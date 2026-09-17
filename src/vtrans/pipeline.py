"""Stage orchestration with resume support.

Every stage writes its result into the job's work directory. Re-running the same
job skips stages whose artefacts already exist, so a crash three hours into a
long video does not cost you the transcription.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import asr as asr_mod
from . import media, segment, separate, subtitles, sync, translate, tts
from .config import Config
from .device import DeviceInfo, describe, detect, free_memory
from .mux import mux
from .utils import (banner, fmt_duration, human_size, media_duration, read_json,
                    require_tool, write_json)

LOG = logging.getLogger("vtrans")

STAGES = ["fetch", "separate", "asr", "segment", "translate", "tts", "subtitles", "mux"]


def job_id(source: str) -> str:
    """Stable id so re-running the same input resumes in the same work dir."""
    name = Path(source).stem if not media.is_url(source) else "url"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name)[:40].strip("-") or "job"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


@dataclass
class StageResult:
    name: str
    note: str          # "" | "reused" | "skipped"
    seconds: float


class Pipeline:
    def __init__(self, cfg: Config, source: str, out_path: Path, *,
                 force_from: Optional[str] = None, glossary_path: Optional[str] = None):
        self.cfg = cfg
        self.source = source
        self.out_path = out_path
        self.glossary_path = glossary_path

        self.models_dir = cfg.resolve_dir("general.models_dir")
        self.work_root = cfg.resolve_dir("general.work_dir")
        self.job_dir = self.work_root / job_id(source)
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.dev: DeviceInfo = detect(cfg.get("general.device", "auto"))
        self.results: List[StageResult] = []
        self._invalidate_from(force_from)

        # Artefact paths
        self.audio_wav = self.job_dir / "audio16k.wav"
        self.asr_json = self.job_dir / "asr.json"
        self.sentences_json = self.job_dir / "sentences.json"
        self.translated_json = self.job_dir / "translated.json"
        self.clips_json = self.job_dir / "clips.json"
        self.dub_wav = self.job_dir / "dub_raw.wav"
        self.final_wav = self.job_dir / "dub_final.wav"
        self.en_srt = self.job_dir / "subtitles.en.srt"
        self.zh_srt = self.job_dir / "subtitles.zh.srt"
        self.ass = self.job_dir / "subtitles.en.ass"

    # -- housekeeping ----------------------------------------------------

    def _invalidate_from(self, stage: Optional[str]) -> None:
        """Delete artefacts so a stage and everything after it re-runs."""
        if not stage:
            return
        if stage not in STAGES:
            raise ValueError(f"--force-from must be one of: {', '.join(STAGES)}")
        order = STAGES.index(stage)
        doomed = {
            "separate": ["background.wav", "demucs"],
            "asr": ["asr.json"],
            "segment": ["sentences.json"],
            "translate": ["translated.json"],
            "tts": ["clips.json", "dub_raw.wav", "dub_final.wav"],
            "subtitles": ["subtitles.en.srt", "subtitles.zh.srt", "subtitles.en.ass"],
        }
        for name in STAGES[order:]:
            for artefact in doomed.get(name, []):
                path = self.job_dir / artefact
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
        LOG.info("Forcing re-run from stage '%s'", stage)

    def _stage(self, index: int, title: str):
        banner(index, len(STAGES), title)
        return time.time()

    def _record(self, name: str, started: float, note: str = "") -> None:
        self.results.append(StageResult(name, note, time.time() - started))

    # -- stages ----------------------------------------------------------

    def run(self) -> Path:
        require_tool("ffmpeg", "Install it with your package manager (apt install ffmpeg).")
        require_tool("ffprobe")

        LOG.info("Device: %s", describe(self.dev))
        LOG.info("Work directory: %s", self.job_dir)

        video = self._stage_fetch()
        duration = media_duration(video)
        LOG.info("Source: %s (%s, %s)", video.name, fmt_duration(duration),
                 human_size(video.stat().st_size))

        background = self._stage_separate()
        asr_segments = self._stage_asr(duration)
        sentences = self._stage_segment(asr_segments)
        sentences = self._stage_translate(sentences)
        clips = self._stage_tts(sentences, duration)
        ass_path, srt_path, zh_srt_path = self._stage_subtitles(sentences, clips, duration)
        final = self._stage_mux(video, background, ass_path, srt_path, zh_srt_path, duration)

        self._report()
        return final

    def _stage_fetch(self) -> Path:
        started = self._stage(1, "Fetching source and extracting audio")
        video = media.fetch_source(self.source, self.job_dir)
        media.extract_audio(video, self.audio_wav,
                            sample_rate=int(self.cfg.get("source.asr_sample_rate", 16000)))
        self._record("fetch", started)
        self.video_path = video
        return video

    def _stage_separate(self) -> Optional[Path]:
        started = self._stage(2, "Separating background music/effects")
        if not self.cfg.get("separate.enabled", False):
            LOG.info("Skipped (enable with --keep-background)")
            self._record("separate", started, note="skipped")
            return None
        # Demucs wants the full-bandwidth audio, not the 16 kHz ASR copy.
        hifi = self.job_dir / "audio48k.wav"
        media.extract_audio(self.video_path, hifi,
                            sample_rate=int(self.cfg.get("source.out_sample_rate", 48000)),
                            channels=2)
        background = separate.separate_background(
            hifi, self.job_dir, self.models_dir, self.dev,
            model=self.cfg.get("separate.model", "htdemucs"),
            segment=int(self.cfg.get("separate.segment", 7)),
        )
        free_memory()
        self._record("separate", started)
        return background

    def _stage_asr(self, duration: float) -> List[asr_mod.ASRSegment]:
        started = self._stage(3, "Transcribing Chinese speech (Whisper)")
        if self.asr_json.exists():
            LOG.info("Reusing existing transcription")
            segments = [asr_mod.ASRSegment.from_dict(d) for d in read_json(self.asr_json)]
            self._record("asr", started, note="reused")
            return segments

        spans = media.plan_chunks(self.audio_wav,
                                  float(self.cfg.get("source.chunk_minutes", 20)))
        LOG.info("Processing %d chunk(s) covering %s", len(spans), fmt_duration(duration))
        chunks = media.split_audio(self.audio_wav, spans, self.job_dir / "chunks",
                                   sample_rate=int(self.cfg.get("source.asr_sample_rate", 16000)))
        segments = asr_mod.transcribe_chunks(chunks, self.cfg, self.dev, self.models_dir)
        write_json(self.asr_json, [s.to_dict() for s in segments])
        free_memory()
        self._record("asr", started)
        return segments

    def _stage_segment(self, asr_segments: List[asr_mod.ASRSegment]) -> List[segment.Sentence]:
        started = self._stage(4, "Assembling sentences")
        if self.sentences_json.exists():
            sentences = [segment.Sentence.from_dict(d) for d in read_json(self.sentences_json)]
            LOG.info("Reusing %d sentences", len(sentences))
            self._record("segment", started, note="reused")
            return sentences

        sentences = segment.build_sentences(
            asr_segments,
            max_chars=int(self.cfg.get("segment.max_chars", 40)),
            max_duration=float(self.cfg.get("segment.max_duration", 8.0)),
            min_duration=float(self.cfg.get("segment.min_duration", 0.7)),
            merge_gap=float(self.cfg.get("segment.merge_gap", 0.35)),
            min_break_chars=int(self.cfg.get("segment.min_break_chars", 4)),
        )
        if not sentences:
            raise RuntimeError(
                "No speech was recognised. Check that the input really contains "
                "Chinese speech, or try --asr-model large-v3 / --no-vad."
            )
        write_json(self.sentences_json, [s.to_dict() for s in sentences])
        self._record("segment", started)
        return sentences

    def _stage_translate(self, sentences: List[segment.Sentence]) -> List[segment.Sentence]:
        started = self._stage(5, "Translating to English")
        if self.translated_json.exists():
            sentences = [segment.Sentence.from_dict(d) for d in read_json(self.translated_json)]
            LOG.info("Reusing %d translations", len(sentences))
            self._record("translate", started, note="reused")
            return sentences

        glossary = translate.load_glossary(self.cfg.get("translate.glossary"), self.glossary_path)

        def progress(done: int, total: int) -> None:
            if done % 40 == 0 or done == total:
                LOG.info("  translated %d/%d unique lines", done, total)

        sentences = translate.translate_sentences(
            sentences, self.cfg, self.dev, self.models_dir,
            glossary=glossary, progress=progress)
        write_json(self.translated_json, [s.to_dict() for s in sentences])
        free_memory()
        self._record("translate", started)
        return sentences

    def _stage_tts(self, sentences: List[segment.Sentence], duration: float) -> List[sync.Clip]:
        started = self._stage(6, "Synthesising English speech and fitting timing")
        if self.clips_json.exists() and self.final_wav.exists():
            clips = [sync.Clip(**d) for d in read_json(self.clips_json)]
            LOG.info("Reusing %d rendered clips", len(clips))
            self._record("tts", started, note="reused")
            return clips

        engine = tts.build_engine(self.cfg, self.models_dir,
                                  device=self.dev.device if self.dev.is_cuda else "cpu")

        def progress(done: int, total: int) -> None:
            LOG.info("  synthesised %d/%d lines", done, total)

        try:
            clips = sync.render_track(sentences, engine, self.cfg, duration,
                                      self.dub_wav, progress=progress)
        finally:
            engine.close()
            free_memory()

        background = self.job_dir / "background.wav"
        sync.mix_with_background(
            self.dub_wav,
            background if (self.cfg.get("separate.enabled") and background.exists()) else None,
            self.final_wav,
            background_gain_db=float(self.cfg.get("separate.background_gain_db", -3.0)),
            loudness_lufs=float(self.cfg.get("sync.loudness_lufs", -16.0)),
            sample_rate=int(self.cfg.get("source.out_sample_rate", 48000)),
            duration=duration,
        )
        write_json(self.clips_json, [c.to_dict() for c in clips])
        self._record("tts", started)
        return clips

    def _stage_subtitles(self, sentences, clips, duration):
        started = self._stage(7, "Building subtitles")
        mode = self.cfg.get("subtitles.mode", "both")
        max_chars = int(self.cfg.get("subtitles.max_line_chars", 42))
        max_lines = int(self.cfg.get("subtitles.max_lines", 2))

        cues = subtitles.build_cues(sentences, clips, use_target=True,
                                    max_line_chars=max_chars, max_lines=max_lines,
                                    total_duration=duration)
        subtitles.write_srt(cues, self.en_srt)
        subtitles.write_ass(
            cues, self.ass,
            font=self.cfg.get("subtitles.font", "DejaVu Sans"),
            size=int(self.cfg.get("subtitles.font_size", 22)),
            outline=int(self.cfg.get("subtitles.outline", 2)),
            shadow=int(self.cfg.get("subtitles.shadow", 0)),
            margin_v=int(self.cfg.get("subtitles.margin_v", 28)),
        )

        zh_path = None
        if self.cfg.get("subtitles.keep_source_srt", True):
            zh_cues = subtitles.build_cues(sentences, None, use_target=False,
                                           max_line_chars=28, max_lines=max_lines,
                                           total_duration=duration)
            zh_path = subtitles.write_srt(zh_cues, self.zh_srt)

        self._record("subtitles", started)
        return (self.ass if mode in ("burn", "both") else None,
                self.en_srt if mode in ("soft", "both") else None,
                zh_path if mode in ("soft", "both") else None)

    def _stage_mux(self, video, background, ass_path, srt_path, zh_srt_path,
                   duration: float) -> Path:
        started = self._stage(8, "Muxing final video")
        mode = self.cfg.get("subtitles.mode", "both")
        final = mux(
            video, self.final_wav, self.out_path,
            ass_path=ass_path, srt_path=srt_path, zh_srt_path=zh_srt_path,
            burn=mode in ("burn", "both"),
            soft=mode in ("soft", "both"),
            video_codec=self.cfg.get("output.video_codec", "auto"),
            crf=int(self.cfg.get("output.crf", 20)),
            preset=self.cfg.get("output.preset", "medium"),
            audio_bitrate=self.cfg.get("output.audio_bitrate", "192k"),
            keep_original_audio=bool(self.cfg.get("output.keep_original_audio", True)),
            duration=duration,
        )
        self._record("mux", started)
        return final

    # -- reporting -------------------------------------------------------

    def _report(self) -> None:
        total = sum(r.seconds for r in self.results)
        LOG.info("")
        LOG.info("Stage timings:")
        for r in self.results:
            suffix = f" ({r.note})" if r.note else ""
            LOG.info("  %-10s %8s%s", r.name, fmt_duration(r.seconds), suffix)
        LOG.info("  %-10s %8s", "total", fmt_duration(total))
        LOG.info("")
        LOG.info("Output:     %s (%s)", self.out_path, human_size(self.out_path.stat().st_size))
        LOG.info("Subtitles:  %s", self.en_srt)
        if not self.cfg.get("general.keep_work", True):
            shutil.rmtree(self.job_dir, ignore_errors=True)
            LOG.info("Work directory removed (general.keep_work: false)")
        else:
            LOG.info("Work files: %s", self.job_dir)
