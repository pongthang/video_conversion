"""Speech recognition with faster-whisper (CTranslate2).

Produces segments carrying absolute timestamps against the *original* media,
even when the audio was processed in chunks.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .device import DeviceInfo, free_memory, whisper_compute_type
from .media import Chunk
from .utils import fmt_duration

LOG = logging.getLogger("vtrans")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ASRSegment":
        return ASRSegment(
            start=d["start"], end=d["end"], text=d["text"],
            words=[Word(**w) for w in d.get("words", [])],
            no_speech_prob=d.get("no_speech_prob", 0.0),
            avg_logprob=d.get("avg_logprob", 0.0),
        )


def resolve_model(model: str, models_dir: Path) -> str:
    """Accept a size name, a HF repo id, or a local directory."""
    local = models_dir / "whisper" / model
    if local.is_dir():
        LOG.info("Using local Whisper model at %s", local)
        return str(local)
    return model


class Transcriber:
    def __init__(self, model_name: str, dev: DeviceInfo, models_dir: Path,
                 compute_type: str = "auto", cpu_threads: int = 0):
        from faster_whisper import WhisperModel

        ctype = whisper_compute_type(dev, compute_type)
        target = resolve_model(model_name, models_dir)
        cache = str(models_dir / "whisper")
        os.makedirs(cache, exist_ok=True)

        LOG.info("Loading Whisper '%s' on %s (compute_type=%s)",
                 model_name, dev.device, ctype)
        try:
            self.model = WhisperModel(
                target,
                device=dev.device,
                device_index=dev.index if dev.is_cuda else 0,
                compute_type=ctype,
                download_root=cache,
                cpu_threads=cpu_threads or (os.cpu_count() or 4),
            )
        except Exception as exc:
            if dev.is_cuda:
                LOG.warning("CUDA load failed (%s); falling back to CPU int8.", exc)
                self.model = WhisperModel(
                    target, device="cpu", compute_type="int8",
                    download_root=cache, cpu_threads=cpu_threads or (os.cpu_count() or 4),
                )
            else:
                raise

    def transcribe_chunk(self, chunk: Chunk, *, language: str = "zh", beam_size: int = 5,
                         vad_filter: bool = True, vad_min_silence_ms: int = 400,
                         condition_on_previous_text: bool = False,
                         word_timestamps: bool = True,
                         initial_prompt: Optional[str] = None) -> List[ASRSegment]:
        segments_iter, info = self.model.transcribe(
            str(chunk.path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms} if vad_filter else None,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )
        LOG.debug("chunk %d: detected language %s (p=%.2f)",
                  chunk.index, info.language, info.language_probability)

        out: List[ASRSegment] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            words = [
                Word(start=w.start + chunk.start, end=w.end + chunk.start, text=w.word)
                for w in (seg.words or []) if w.start is not None and w.end is not None
            ]
            out.append(ASRSegment(
                start=seg.start + chunk.start,
                end=seg.end + chunk.start,
                text=text,
                words=words,
                no_speech_prob=getattr(seg, "no_speech_prob", 0.0) or 0.0,
                avg_logprob=getattr(seg, "avg_logprob", 0.0) or 0.0,
            ))
        return out

    def close(self) -> None:
        self.model = None
        free_memory()


_PUNCT = "，。！？、；：,.!?;: \t"


def _normalise(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT)


def _echoed_elsewhere(seg: ASRSegment, others: List[ASRSegment]) -> bool:
    """True if this segment's text also appears at an unrelated point in time."""
    import difflib

    text = _normalise(seg.text)
    if len(text) < 4:
        return False
    for other in others:
        if other is seg or abs(other.start - seg.start) < 10.0:
            continue
        candidate = _normalise(other.text)
        if len(candidate) < 4:
            continue
        if text in candidate or candidate in text:
            return True
        if difflib.SequenceMatcher(None, text, candidate).ratio() >= 0.8:
            return True
    return False


def _drop_boundary_artifacts(per_chunk: List[List[ASRSegment]], chunks: List[Chunk],
                             lead_window: float = 6.0, tail_window: float = 2.0
                             ) -> List[ASRSegment]:
    """Merge chunk transcripts, dropping Whisper's echoes at the seams.

    Whisper often invents a line just after the start of an audio chunk, and it
    is usually a verbatim echo of speech from elsewhere in the recording. Two
    conditions must both hold before anything is discarded: the segment sits
    next to a seam and overlaps its neighbour in time (real segments never
    overlap), and its text is repeated somewhere else. Requiring the echo means
    unique content is never thrown away, which matters because the alternative
    - guessing from confidence scores - silently deletes real speech.
    """
    flagged: List[tuple] = []
    for chunk, segments in zip(chunks, per_chunk):
        for seg in segments:
            near_seam = ((seg.start - chunk.start) <= lead_window
                         or (chunk.end - seg.end) <= tail_window)
            flagged.append((seg, near_seam))
    flagged.sort(key=lambda pair: pair[0].start)
    all_segments = [seg for seg, _ in flagged]

    keep = [True] * len(flagged)
    for i in range(len(flagged) - 1):
        if not keep[i]:
            continue
        cur, cur_seam = flagged[i]
        nxt, nxt_seam = flagged[i + 1]
        if not (cur_seam or nxt_seam):
            continue
        overlap = cur.end - nxt.start
        shorter = min(cur.end - cur.start, nxt.end - nxt.start)
        if overlap <= 0.2 or shorter <= 0 or overlap < shorter * 0.3:
            continue

        for index, (seg, seam) in ((i, (cur, cur_seam)), (i + 1, (nxt, nxt_seam))):
            if seam and _echoed_elsewhere(seg, all_segments):
                keep[index] = False
                LOG.debug("Dropping chunk-seam echo at %.2f: %s", seg.start, seg.text)
                break

    dropped = keep.count(False)
    if dropped:
        LOG.info("Removed %d echoed line(s) at chunk boundaries", dropped)
    return [seg for (seg, _), ok in zip(flagged, keep) if ok]


def transcribe_chunks(chunks: List[Chunk], cfg, dev: DeviceInfo, models_dir: Path,
                      progress=None) -> List[ASRSegment]:
    tr = Transcriber(
        model_name=cfg.get("asr.model"),
        dev=dev,
        models_dir=models_dir,
        compute_type=cfg.get("asr.compute_type", "auto"),
    )
    try:
        per_chunk: List[List[ASRSegment]] = []
        for chunk in chunks:
            LOG.info("Transcribing chunk %d/%d  (%s -> %s)",
                     chunk.index + 1, len(chunks),
                     fmt_duration(chunk.start), fmt_duration(chunk.end))
            segs = tr.transcribe_chunk(
                chunk,
                language=cfg.get("asr.language", "zh"),
                beam_size=cfg.get("asr.beam_size", 5),
                vad_filter=cfg.get("asr.vad_filter", True),
                vad_min_silence_ms=cfg.get("asr.vad_min_silence_ms", 400),
                condition_on_previous_text=cfg.get("asr.condition_on_previous_text", False),
                word_timestamps=cfg.get("asr.word_timestamps", True),
                initial_prompt=cfg.get("asr.initial_prompt"),
            )
            LOG.info("  -> %d segments", len(segs))
            per_chunk.append(segs)
            if progress:
                progress(chunk.index + 1, len(chunks))
    finally:
        tr.close()

    if len(per_chunk) == 1:
        return sorted(per_chunk[0], key=lambda s: s.start)
    return _drop_boundary_artifacts(per_chunk, chunks)
