"""Turn raw ASR output into sentence-level units.

Whisper emits arbitrary segments that often end mid-sentence. Translating those
directly gives fragmented, low-quality English. Re-cutting the word stream on
Chinese sentence punctuation gives the MT model whole sentences and gives the
subtitles natural line breaks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

from .asr import ASRSegment, Word

LOG = logging.getLogger("vtrans")

# Strong sentence terminators, then weaker clause breaks used only when a unit
# has grown too long to be a comfortable subtitle.
SENTENCE_END = "。！？!?…"
CLAUSE_END = "，,、；;：:"
CLOSERS = "”’\"')）】》」』"

# Common Whisper hallucinations on Chinese media (subscribe/like banners burned
# into training data). These appear during music or silence and are dropped.
HALLUCINATION_PATTERNS = [
    re.compile(r"^(请不吝点赞|订阅|转发|打赏支持明镜|明镜与点点栏目)"),
    re.compile(r"^(字幕由|字幕制作|本字幕|由.{0,10}字幕组)"),
    re.compile(r"^(谢谢大家|谢谢观看|感谢观看|謝謝觀看)[。！!.]?$"),
    re.compile(r"^(MING PAO|Amara\.org|字幕志愿者)", re.IGNORECASE),
]


@dataclass
class Sentence:
    index: int
    start: float
    end: float
    source: str                 # Chinese text
    target: str = ""            # English translation, filled in later
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Sentence":
        return Sentence(
            index=d["index"], start=d["start"], end=d["end"],
            source=d["source"], target=d.get("target", ""),
            words=[Word(**w) for w in d.get("words", [])],
        )


def _is_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if any(p.search(stripped) for p in HALLUCINATION_PATTERNS):
        return True
    # A single character repeated many times ("啊啊啊啊啊啊啊...").
    if len(stripped) >= 8 and len(set(stripped)) <= 2:
        return True
    return False


# Break strength reported by the word stream.
BREAK_NONE, BREAK_SPACE, BREAK_SEGMENT = 0, 1, 2


def _flatten_words(segments: List[ASRSegment]) -> List[Tuple[Word, int]]:
    """One continuous word stream, tagging where Whisper saw a boundary.

    In fast dialogue Whisper often emits no punctuation at all, but it still
    cuts segments at utterance boundaries and separates turns with a space.
    Those two signals are the only reliable boundaries available, and without
    them a whole exchange collapses into one run-on line that the translator
    then silently truncates.
    """
    stream: List[Tuple[Word, int]] = []
    for seg in segments:
        if _is_hallucination(seg.text):
            LOG.debug("Dropping likely hallucination at %.2f: %s", seg.start, seg.text)
            continue

        words = list(seg.words)
        if not words:
            # No word timestamps: distribute characters evenly across the segment.
            chars = list(seg.text.strip())
            if not chars:
                continue
            step = max(1e-3, (seg.end - seg.start) / len(chars))
            words = [Word(start=seg.start + i * step, end=seg.start + (i + 1) * step, text=c)
                     for i, c in enumerate(chars)]

        for i, word in enumerate(words):
            if i == len(words) - 1:
                kind = BREAK_SEGMENT
            elif word.text.endswith(" "):
                kind = BREAK_SPACE
            else:
                kind = BREAK_NONE
            stream.append((word, kind))
    return stream


def _flush(buf: List[Word], out: List[Sentence]) -> None:
    text = "".join(w.text for w in buf).strip()
    if not text or _is_hallucination(text):
        buf.clear()
        return
    out.append(Sentence(index=len(out), start=buf[0].start, end=buf[-1].end,
                        source=text, words=list(buf)))
    buf.clear()


def build_sentences(segments: List[ASRSegment], *, max_chars: int = 40,
                    max_duration: float = 8.0, min_duration: float = 0.7,
                    merge_gap: float = 0.35, min_break_chars: int = 4) -> List[Sentence]:
    stream = _flatten_words(segments)
    if not stream:
        return []

    sentences: List[Sentence] = []
    buf: List[Word] = []

    for i, (word, kind) in enumerate(stream):
        buf.append(word)
        text = "".join(w.text for w in buf).strip()
        if not text:
            continue

        next_word = stream[i + 1][0] if i + 1 < len(stream) else None
        ends_sentence = text[-1] in SENTENCE_END
        # Keep closing quotes/brackets attached to the sentence they close.
        if ends_sentence and next_word:
            nxt = next_word.text.strip()
            if nxt and nxt[0] in CLOSERS:
                ends_sentence = False

        hard_limit = len(text) >= max_chars or (buf[-1].end - buf[0].start) >= max_duration
        # Long enumerations carry commas but no full stop; split them so no
        # single line grows past what the translator handles well.
        clause_break = text[-1] in CLAUSE_END and len(text) >= max_chars * 0.6
        utterance_break = (kind == BREAK_SEGMENT
                           or (kind == BREAK_SPACE and len(text) >= min_break_chars))

        if ends_sentence or hard_limit or clause_break or utterance_break:
            _flush(buf, sentences)

    _flush(buf, sentences)

    merged = _merge_short(sentences, min_duration=min_duration, merge_gap=merge_gap,
                          max_chars=max_chars, max_duration=max_duration)
    for i, sentence in enumerate(merged):
        sentence.index = i
    LOG.info("Assembled %d sentences from %d ASR segments", len(merged), len(segments))
    return merged


def _merge_short(sentences: List[Sentence], *, min_duration: float, merge_gap: float,
                 max_chars: int, max_duration: float) -> List[Sentence]:
    """Glue fragments like "对。" onto their neighbour when they sit right next to it."""
    if not sentences:
        return []
    out: List[Sentence] = [sentences[0]]
    for cur in sentences[1:]:
        prev = out[-1]
        gap = cur.start - prev.end
        combined_chars = len(prev.source) + len(cur.source)
        combined_dur = cur.end - prev.start
        too_short = prev.duration < min_duration or cur.duration < min_duration
        ends_sentence = bool(prev.source) and prev.source[-1] in SENTENCE_END
        if (gap <= merge_gap and too_short and not ends_sentence
                and combined_chars <= max_chars
                and combined_dur <= max_duration):
            prev.source = (prev.source + cur.source).strip()
            prev.end = cur.end
            prev.words.extend(cur.words)
        else:
            out.append(cur)
    return out


def strip_for_tts(text: str) -> str:
    """Clean an English line before it is handed to the TTS engine."""
    text = re.sub(r"\s+", " ", text).strip()
    # Full-width punctuation sometimes survives translation; TTS reads it badly.
    table = {"，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
             "（": "(", "）": ")", "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text.strip()
