"""Stage orchestration with resume support.

Every stage writes its result into the job's work directory. Re-running the same
job skips stages whose artefacts already exist, so a crash three hours into a
long video does not cost you the transcription.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import asr as asr_mod
from . import media, segment, separate, subtitles, sync, translate, tts
from .config import Config
from .device import DeviceInfo, describe, detect, free_memory, free_vram_gb
from .gpulock import maybe_lock
from .fallback import FallbackPolicy
from .mux import mux
from .progress import Cancelled, CancelToken, NullReporter, Reporter, estimate
from .utils import (banner, fmt_duration, human_size, media_duration, read_json,
                    require_tool, write_json)

LOG = logging.getLogger("vtrans")

STAGES = ["fetch", "separate", "asr", "segment", "translate", "tts", "subtitles", "mux"]


def job_id(source: str) -> str:
    """Stable id so re-running the same input resumes in the same work dir.

    The id has to identify the *file*, not the spelling of its path. Qt's file
    dialog hands back "D:/dir/clip.mp4" while a shell hands back
    "D:\\dir\\clip.mp4", and hashing those raw strings gave the same video two
    different work directories: the GUI then reused its own stale artefacts and
    a command line --force-from cleaned a directory the GUI never looked at.

    So the path is resolved first, separators are normalised, and on Windows
    the result is lowercased because its filesystem is case-insensitive.
    """
    if media.is_url(source):
        name, key = "url", source.strip()
    else:
        path = Path(source)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):     # broken symlink, or path too long
            resolved = path.absolute()
        name = resolved.stem
        key = str(resolved).replace("\\", "/")
        if os.name == "nt":
            key = key.lower()

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name)[:40].strip("-") or "job"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


@dataclass
class StageResult:
    name: str
    note: str          # "" | "reused" | "skipped"
    seconds: float


class Pipeline:
    def __init__(self, cfg: Config, source: str, out_path: Path, *,
                 force_from: Optional[str] = None, glossary_path: Optional[str] = None,
                 reporter: Optional[Reporter] = None,
                 token: Optional[CancelToken] = None,
                 cpu_fallback: bool = True,
                 gpu_lock: bool = True):
        self.cfg = cfg
        self.source = source
        self.out_path = out_path
        self.glossary_path = glossary_path
        # The CLI passes nothing and gets a reporter whose methods are no-ops,
        # so the stage bodies below need no "if reporter" branching.
        self.reporter = reporter or NullReporter()
        self.token = token
        if token is not None and self.reporter.token is None:
            self.reporter.token = token

        self.models_dir = cfg.resolve_dir("general.models_dir")
        self.work_root = cfg.resolve_dir("general.work_dir")
        self.job_dir = self.work_root / job_id(source)
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.dev: DeviceInfo = detect(cfg.get("general.device", "auto"))
        self.gpu_lock = gpu_lock
        self.fallback = FallbackPolicy(enabled=cpu_fallback and self.dev.is_cuda,
                                       reporter=self.reporter)
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
        self.reporter.check_cancelled()
        banner(index, len(STAGES), title)
        self.reporter.stage_started(STAGES[index - 1], title)
        return time.time()

    def _record(self, name: str, started: float, note: str = "") -> None:
        self.results.append(StageResult(name, note, time.time() - started))
        self.reporter.stage_finished(name)

    def _duration_hint(self) -> float:
        """Source length before fetching, for weighting. 0 when unknowable.

        Only the mux weight depends on it, and only past the one-hour mark, so
        a URL whose length is not yet known simply gets the default weighting.
        """
        candidate = Path(self.source)
        if candidate.is_file():
            try:
                return media_duration(candidate)
            except Exception:  # noqa: BLE001 - weighting is advisory
                return 0.0
        return 0.0

    # Config keys whose value changes the output of each cached stage. Reuse is
    # only valid while these are unchanged: without this, switching the voice
    # from female to male, or the preset from Balanced to Best, silently
    # replayed the audio rendered under the old settings, because the check was
    # only ever "does clips.json exist".
    STAGE_INPUTS = {
        "asr": ["asr.model", "asr.language", "asr.compute_type", "asr.beam_size",
                "asr.vad_filter", "asr.vad_min_silence_ms",
                "asr.condition_on_previous_text", "asr.word_timestamps",
                "asr.initial_prompt", "source.chunk_minutes"],
        "segment": ["segment.max_chars", "segment.max_duration", "segment.min_duration",
                    "segment.merge_gap", "segment.min_break_chars"],
        "translate": ["translate.backend", "translate.model", "translate.src_lang",
                      "translate.tgt_lang", "translate.num_beams",
                      "translate.max_new_tokens", "translate.glossary"],
        "tts": ["tts.backend", "tts.voice", "tts.kokoro_voice", "tts.length_scale",
                "tts.noise_scale", "tts.noise_w", "tts.sentence_silence",
                "sync.max_speedup", "sync.min_length_scale", "sync.gap_usage",
                "sync.allow_overlap", "sync.loudness_lufs",
                "separate.enabled", "separate.background_gain_db"],
    }

    def _fingerprint_path(self, stage: str) -> Path:
        return self.job_dir / f"{stage}.inputs.json"

    def _fingerprint(self, stage: str) -> dict:
        return {key: self.cfg.get(key) for key in self.STAGE_INPUTS.get(stage, [])}

    def _may_reuse(self, stage: str) -> bool:
        """Whether the cached artefacts for `stage` were made with these settings.

        A missing fingerprint file means the artefacts predate this check, so
        they are accepted: re-running an expensive transcription because the
        bookkeeping is new would be worse than the small risk of a stale reuse.
        """
        path = self._fingerprint_path(stage)
        if not path.exists():
            return True
        try:
            previous = read_json(path)
        except (OSError, ValueError):
            return True
        current = self._fingerprint(stage)
        if previous == current:
            return True

        changed = [k for k in current if previous.get(k) != current.get(k)]
        LOG.info("Settings changed since the cached %s (%s); re-running it",
                 stage, ", ".join(changed))
        return False

    def _save_fingerprint(self, stage: str) -> None:
        write_json(self._fingerprint_path(stage), self._fingerprint(stage))

    def _configure_progress(self, duration: float) -> None:
        """Size the overall bar once the source duration and device are known."""
        self.reporter.configure(estimate(
            STAGES,
            on_cuda=self.dev.is_cuda,
            separating=bool(self.cfg.get("separate.enabled", False)),
            is_url=media.is_url(self.source),
            duration=duration,
        ))

    # -- stages ----------------------------------------------------------

    def run(self) -> Path:
        require_tool("ffmpeg", "Install it with your package manager (apt install ffmpeg).")
        require_tool("ffprobe")

        with maybe_lock(self.work_root, self.dev.is_cuda, self.gpu_lock):
            return self._run_stages()

    def _run_stages(self) -> Path:

        LOG.info("Device: %s", describe(self.dev))
        if self.dev.is_cuda:
            # Stages load one model at a time and free it before the next, so
            # this figure is what each stage individually has to work with.
            LOG.info("Free VRAM: %.1f GB (one model resident at a time)",
                     free_vram_gb(self.dev))
        LOG.info("Work directory: %s", self.job_dir)

        # Weights must be in place before the first stage_started call,
        # otherwise that stage contributes 0 and the bar tops out short of 1.0.
        self._configure_progress(self._duration_hint())

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

        self.reporter.stage_progress(1.0)
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
        background = self.fallback.run_stage(
            "separate", self.dev,
            lambda dev: separate.separate_background(
                hifi, self.job_dir, self.models_dir, dev,
                model=self.cfg.get("separate.model", "htdemucs"),
                segment=int(self.cfg.get("separate.segment", 7)),
            ))
        free_memory()
        self._record("separate", started)
        return background

    def _stage_asr(self, duration: float) -> List[asr_mod.ASRSegment]:
        started = self._stage(3, "Transcribing Chinese speech (Whisper)")
        if self.asr_json.exists() and self._may_reuse("asr"):
            LOG.info("Reusing existing transcription")
            segments = [asr_mod.ASRSegment.from_dict(d) for d in read_json(self.asr_json)]
            self._record("asr", started, note="reused")
            return segments

        spans = media.plan_chunks(self.audio_wav,
                                  float(self.cfg.get("source.chunk_minutes", 20)))
        LOG.info("Processing %d chunk(s) covering %s", len(spans), fmt_duration(duration))
        chunks = media.split_audio(self.audio_wav, spans, self.job_dir / "chunks",
                                   sample_rate=int(self.cfg.get("source.asr_sample_rate", 16000)))

        def progress(done: int, total: int) -> None:
            self.reporter.stage_progress(done / max(1, total))
            self.reporter.check_cancelled()

        segments = self.fallback.run_stage(
            "asr", self.dev,
            lambda dev: asr_mod.transcribe_chunks(chunks, self.cfg, dev, self.models_dir,
                                                  progress=progress))
        write_json(self.asr_json, [s.to_dict() for s in segments])
        self._save_fingerprint("asr")
        free_memory()
        self._record("asr", started)
        return segments

    def _stage_segment(self, asr_segments: List[asr_mod.ASRSegment]) -> List[segment.Sentence]:
        started = self._stage(4, "Assembling sentences")
        if self.sentences_json.exists() and self._may_reuse("segment"):
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
        self._save_fingerprint("segment")
        self._record("segment", started)
        return sentences

    def _stage_translate(self, sentences: List[segment.Sentence]) -> List[segment.Sentence]:
        started = self._stage(5, "Translating to English")
        if self.translated_json.exists() and self._may_reuse("translate"):
            sentences = [segment.Sentence.from_dict(d) for d in read_json(self.translated_json)]
            LOG.info("Reusing %d translations", len(sentences))
            self._record("translate", started, note="reused")
            return sentences

        glossary = translate.load_glossary(self.cfg.get("translate.glossary"), self.glossary_path)

        def progress(done: int, total: int) -> None:
            if done % 40 == 0 or done == total:
                LOG.info("  translated %d/%d unique lines", done, total)
            self.reporter.stage_progress(done / max(1, total))
            self.reporter.check_cancelled()

        sentences = self.fallback.run_stage(
            "translate", self.dev,
            lambda dev: translate.translate_sentences(
                sentences, self.cfg, dev, self.models_dir,
                glossary=glossary, progress=progress))
        write_json(self.translated_json, [s.to_dict() for s in sentences])
        self._save_fingerprint("translate")
        free_memory()
        self._record("translate", started)
        return sentences

    def _stage_tts(self, sentences: List[segment.Sentence], duration: float) -> List[sync.Clip]:
        started = self._stage(6, "Synthesising English speech and fitting timing")
        if self.clips_json.exists() and self.final_wav.exists() and self._may_reuse("tts"):
            clips = [sync.Clip(**d) for d in read_json(self.clips_json)]
            LOG.info("Reusing %d rendered clips", len(clips))
            self._record("tts", started, note="reused")
            return clips

        def progress(done: int, total: int) -> None:
            LOG.info("  synthesised %d/%d lines", done, total)
            # TTS is followed by mixing and loudness normalisation, so leave
            # headroom at the top of the stage rather than reaching 1.0 here.
            self.reporter.stage_progress(0.9 * done / max(1, total))
            self.reporter.check_cancelled()

        def render(dev: DeviceInfo):
            engine = tts.build_engine(self.cfg, self.models_dir,
                                      device=dev.device if dev.is_cuda else "cpu")
            try:
                return sync.render_track(sentences, engine, self.cfg, duration,
                                         self.dub_wav, progress=progress)
            finally:
                engine.close()
                free_memory()

        clips = self.fallback.run_stage("tts", self.dev, render)

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
        self._save_fingerprint("tts")
        self._record("tts", started)
        return clips

    def _stage_subtitles(self, sentences, clips, duration):
        started = self._stage(7, "Building subtitles")
        mode = self.cfg.get("subtitles.mode", "both")
        max_chars = int(self.cfg.get("subtitles.max_line_chars", 48))
        max_lines = int(self.cfg.get("subtitles.max_lines", 2))

        cues = subtitles.build_cues(sentences, clips, use_target=True,
                                    max_line_chars=max_chars, max_lines=max_lines,
                                    total_duration=duration)
        subtitles.write_srt(cues, self.en_srt)
        subtitles.write_ass(
            cues, self.ass,
            font=self.cfg.get("subtitles.font", "DejaVu Sans"),
            size=int(self.cfg.get("subtitles.font_size", 18)),
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
        def on_progress(fraction: float) -> None:
            self.reporter.stage_progress(fraction)

        final = mux(
            video, self.final_wav, self.out_path,
            on_progress=on_progress,
            should_cancel=(lambda: self.reporter.cancelled),
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
        if self.fallback.demoted:
            LOG.warning("%s", self.fallback.summary())
        LOG.info("")
        LOG.info("Output:     %s (%s)", self.out_path, human_size(self.out_path.stat().st_size))
        LOG.info("Subtitles:  %s", self.en_srt)
        if not self.cfg.get("general.keep_work", True):
            shutil.rmtree(self.job_dir, ignore_errors=True)
            LOG.info("Work directory removed (general.keep_work: false)")
        else:
            LOG.info("Work files: %s", self.job_dir)
