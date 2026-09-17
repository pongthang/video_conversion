"""Chinese -> English machine translation.

Backends:
  nllb  - facebook/nllb-200-distilled-600M (default; good quality, ~2.5 GB)
  opus  - Helsinki-NLP/opus-mt-zh-en (tiny and fast, lower quality)
  none  - passthrough, keeps the Chinese text (useful for testing timing)

Translation is sentence-level by design: the sentences were assembled upstream
so the model sees complete thoughts rather than Whisper's arbitrary cuts.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .device import DeviceInfo, free_memory, free_vram_gb, torch_dtype
from .segment import Sentence
from .utils import chunked, hf_cache_dir

LOG = logging.getLogger("vtrans")

BACKEND_DEFAULT_MODELS = {
    "nllb": "auto",
    "opus": "Helsinki-NLP/opus-mt-zh-en",
}

# Largest first. The pipeline loads exactly one model at a time and frees it
# before the next stage, so the whole card is available here - measured on a
# 4 GB GTX 1650: Whisper large-v3 peaks at 1.9 GB and is fully released before
# translation starts. min_free_gb includes room for beam-search activations.
NLLB_TIERS = [
    ("facebook/nllb-200-distilled-1.3B", 3.4, 4),
    ("facebook/nllb-200-distilled-600M", 1.6, 8),
]


def _collapse_repeats(text: str) -> str:
    """NMT models occasionally loop; drop immediate word/phrase repetitions."""
    text = re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\b[\w' ]{4,40}?\b)(\s*\1){2,}", r"\1", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def _tidy(text: str) -> str:
    text = _collapse_repeats(text.strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Sentences split on a Chinese comma can come back with dangling punctuation.
    text = re.sub(r"^[\s,;:.、，。；：]+", "", text)
    text = re.sub(r"[\s,;:、，；：]+$", "", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def looks_degenerate(text: str) -> bool:
    """Spot the collapse mode where beam search loops a phrase instead of translating.

    NLLB does this on short, list-like inputs: "白菜,芦笋,土豆。" comes back as
    "It's not just me, it's me." Greedy decoding usually recovers such lines, so
    they are worth detecting and retrying.
    """
    words = re.findall(r"[\w']+", text.lower())
    if not words:
        return True
    if len(words) <= 15:
        counts: Dict[str, int] = {}
        for word in words:
            if len(word) >= 3:
                counts[word] = counts.get(word, 0) + 1
        if counts and max(counts.values()) >= 3:
            return True
    # A short, genuine translation almost never reuses words. A low
    # type-token ratio is the clearest signal that decoding went in circles.
    if 4 <= len(words) <= 15 and len(set(words)) / len(words) <= 0.7:
        return True
    return False


def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    for src, dst in glossary.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text


WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _is_cached(repo_id: str, models_dir: Path) -> bool:
    """True only if the *weights* are fully downloaded.

    Checking that the snapshot directory merely exists is not enough: the small
    config and tokenizer files arrive first, so a partially downloaded model
    looks complete for the several minutes its multi-GB weights are still in
    flight. Selecting it then would stall the run on the very download this
    check is meant to avoid.
    """
    cache = Path(hf_cache_dir(models_dir))
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    if not folder.is_dir():
        return False

    # A blob still being fetched leaves a .incomplete file behind.
    blobs = folder / "blobs"
    if blobs.is_dir() and any(blobs.glob("*.incomplete")):
        return False

    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    for rev in snapshots.iterdir():
        if not rev.is_dir():
            continue
        for entry in rev.iterdir():
            if not entry.name.endswith(WEIGHT_SUFFIXES):
                continue
            try:
                # resolve() follows the symlink into blobs/; a missing target
                # or a stub-sized file means the download never completed.
                if entry.resolve().stat().st_size > 1_000_000:
                    return True
            except OSError:
                continue
    return False


def select_nllb_model(dev: DeviceInfo, models_dir: Path) -> Tuple[str, Optional[int]]:
    """Pick the largest NLLB that fits free VRAM and is already downloaded.

    Returns (model_name, suggested_batch_size). A bigger model is a much larger
    quality win than any decoding tweak, so it is worth spending spare VRAM on -
    but only when it is already on disk, so a run never stalls on a multi-GB
    download it did not ask for.
    """
    free = free_vram_gb(dev) if dev.is_cuda else 0.0
    smallest = NLLB_TIERS[-1]

    if not dev.is_cuda:
        LOG.info("Translating on CPU with %s", smallest[0])
        return smallest[0], smallest[2]

    for name, need, batch in NLLB_TIERS:
        if free < need:
            continue
        if not _is_cached(name, models_dir):
            if name != smallest[0]:
                LOG.info("%.1f GB VRAM free would fit %s, but it is not downloaded. "
                         "Fetch it once with:  python -m vtrans.download "
                         "--translate-model %s", free, name, name)
            continue
        if name != smallest[0]:
            LOG.info("%.1f GB VRAM free - using the larger %s for better quality",
                     free, name)
        return name, batch

    return smallest[0], smallest[2]


class Translator:
    """Wraps a seq2seq HF model with NLLB's language-token handling."""

    def __init__(self, backend: str, model_name: str, dev: DeviceInfo, models_dir: Path,
                 src_lang: str = "zho_Hans", tgt_lang: str = "eng_Latn"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.backend = backend
        self.dev = dev
        self.torch = torch
        cache = hf_cache_dir(models_dir)

        LOG.info("Loading translation model '%s' on %s", model_name, dev.device)
        tok_kwargs = {"cache_dir": cache}
        if backend == "nllb":
            tok_kwargs["src_lang"] = src_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=cache, torch_dtype=torch_dtype(dev),
        )
        self.model.to(dev.device if dev.is_cuda else "cpu")
        self.model.eval()

        self.forced_bos_token_id = None
        if backend == "nllb":
            self.forced_bos_token_id = self._lang_token_id(tgt_lang)

    def _lang_token_id(self, lang_code: str) -> int:
        """NLLB language tokens; the lookup API changed across transformers versions."""
        tok = self.tokenizer
        mapping = getattr(tok, "lang_code_to_id", None)
        if isinstance(mapping, dict) and lang_code in mapping:
            return mapping[lang_code]
        token_id = tok.convert_tokens_to_ids(lang_code)
        unk = getattr(tok, "unk_token_id", None)
        if token_id is None or token_id == unk:
            raise RuntimeError(
                f"Target language code '{lang_code}' is not known to this tokenizer. "
                "Use an NLLB FLORES-200 code such as 'eng_Latn'."
            )
        return token_id

    @property
    def _device(self) -> str:
        return self.dev.device if self.dev.is_cuda else "cpu"

    def translate_batch(self, texts: List[str], *, num_beams: int = 4,
                        max_new_tokens: int = 256) -> List[str]:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(self._device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "no_repeat_ngram_size": 4,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        if self.forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = self.forced_bos_token_id
        with self.torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        free_memory()


def load_glossary(cfg_glossary: Dict[str, str] | None, path: str | None) -> Dict[str, str]:
    glossary = dict(cfg_glossary or {})
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            glossary.update(json.load(fh))
    if glossary:
        LOG.info("Loaded %d glossary entries", len(glossary))
    return glossary


def translate_sentences(sentences: List[Sentence], cfg, dev: DeviceInfo, models_dir: Path,
                        glossary: Dict[str, str] | None = None,
                        progress=None) -> List[Sentence]:
    backend = cfg.get("translate.backend", "nllb")
    glossary = glossary or {}

    if backend == "none":
        LOG.warning("translate.backend=none - keeping the Chinese text")
        for s in sentences:
            s.target = s.source
        return sentences

    model_name = cfg.get("translate.model") or BACKEND_DEFAULT_MODELS[backend]
    suggested_batch = None
    if backend == "nllb" and model_name in ("auto", "", None):
        model_name, suggested_batch = select_nllb_model(dev, models_dir)

    def _load(name: str) -> Translator:
        return Translator(
            backend=backend, model_name=name, dev=dev, models_dir=models_dir,
            src_lang=cfg.get("translate.src_lang", "zho_Hans"),
            tgt_lang=cfg.get("translate.tgt_lang", "eng_Latn"),
        )

    try:
        tr = _load(model_name)
    except (RuntimeError, MemoryError) as exc:
        fallback = NLLB_TIERS[-1][0]
        if backend != "nllb" or model_name == fallback or "out of memory" not in str(exc).lower():
            raise
        LOG.warning("%s did not fit in VRAM (%s); falling back to %s",
                    model_name, str(exc)[:60], fallback)
        free_memory()
        model_name, suggested_batch = fallback, NLLB_TIERS[-1][2]
        tr = _load(model_name)

    # A larger model needs a smaller batch to leave room for beam-search state.
    batch_size = int(suggested_batch or cfg.get("translate.batch_size", 8))
    num_beams = int(cfg.get("translate.num_beams", 4))
    max_new_tokens = int(cfg.get("translate.max_new_tokens", 256))

    # Identical lines are common in speech ("对", "好的"); translate each once.
    cache: Dict[str, str] = {}
    unique = list(dict.fromkeys(s.source for s in sentences))

    try:
        done = 0
        retried = 0
        for batch in chunked(unique, batch_size):
            try:
                results = tr.translate_batch(batch, num_beams=num_beams,
                                             max_new_tokens=max_new_tokens)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                LOG.warning("Translation OOM at batch size %d; retrying one by one", len(batch))
                free_memory()
                results = [tr.translate_batch([t], num_beams=1,
                                              max_new_tokens=max_new_tokens)[0] for t in batch]
            for src, out in zip(batch, results):
                text = _tidy(out)
                if looks_degenerate(text):
                    retry = _tidy(tr.translate_batch([src], num_beams=1,
                                                     max_new_tokens=max_new_tokens)[0])
                    LOG.debug("Retried degenerate translation of %r: %r -> %r",
                              src, text, retry)
                    if not looks_degenerate(retry):
                        text = retry
                        retried += 1
                cache[src] = apply_glossary(text, glossary)
            done += len(batch)
            if progress:
                progress(done, len(unique))
            LOG.debug("translated %d/%d unique lines", done, len(unique))
    finally:
        tr.close()

    if retried:
        LOG.info("Re-translated %d line(s) that decoded degenerately", retried)

    for s in sentences:
        s.target = cache.get(s.source, "")
    empty = sum(1 for s in sentences if not s.target)
    if empty:
        LOG.warning("%d sentences produced an empty translation", empty)
    return sentences
