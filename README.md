# Chinese → English video dubbing pipeline

Takes a Chinese-language video (local file or YouTube URL) and produces an
English-dubbed video with burned-in English captions. Everything runs locally
with open-source models — no API keys, no per-minute billing.

**Desktop app (Ubuntu)**

```
./setup.sh              # once per machine: venv, dependencies, models, GUI
./install_desktop.sh    # adds it to the applications menu
./run_gui.sh            # or start it directly
```

The window takes a video file, a URL or a drag-and-drop, offers a male or
female voice, shows what each model needs against the GPU it detected, and
previews the captions on a frame of your own video so the size and position can
be judged before committing to a conversion. It shows plainly whether the run
is on the GPU or the CPU, because the difference is large.

`./setup.sh --no-gui` installs the command line only.

**Command line**

```
./setup.sh                                    # once per machine
./convert_video.sh input.mp4 -out output.mp4  # per video
```

**Windows**

A desktop installer is in [windows/](windows/) — written but not yet built,
since building it needs a Windows machine. See [windows/README.md](windows/README.md).

---

## What it does

```
 source video ──▶ audio ──▶ [Demucs] ──▶ background music/SFX ──────────┐
                     │                                                  │
                     └──▶ Whisper (zh) ──▶ sentences ──▶ NLLB (zh→en)    │
                                                            │           │
                                             ┌──────────────┴───────┐   │
                                             ▼                      ▼   ▼
                                     Piper TTS + timing fit      ASS/SRT captions
                                             │                      │
                                             └────────▶ ffmpeg mux ◀─┘
                                                            │
                                                            ▼
                                                        output.mp4
```

| Stage | Model / tool | Notes |
|---|---|---|
| Download | `yt-dlp` | capped at 1080p |
| Separation *(optional)* | Demucs `htdemucs` | keeps the original music under the dub |
| Transcription | faster-whisper `large-v3` | CTranslate2, int8_float16 on small GPUs |
| Sentence assembly | rule-based | re-cuts on Chinese sentence punctuation |
| Translation | NLLB-200-distilled-600M | `opus-mt-zh-en` available as a light option |
| Speech | Piper (Kokoro optional) | MIT / Apache-2.0, fully offline |
| Timing | Piper `length_scale` + rubberband | fits each line to its slot |
| Muxing | ffmpeg + libass | burned-in and/or soft subtitles |

The output `.mp4` contains:

- **video** with English captions burned in,
- **audio track 1** — English dub (default),
- **audio track 2** — original Chinese (not default).

Alongside it, `work/<job>/` keeps `subtitles.en.srt`, `subtitles.zh.srt` and the
intermediate JSON for every stage.

### Subtitles: burned in, embedded, or both

`--subtitles both` (the default) burns the captions into the picture. It does
**not** also embed a subtitle track in an MP4: the MP4 muxer auto-enables the
first subtitle track no matter what disposition it is given, so the player would
render the same text a second time on top of the burned-in captions. Matroska
respects the flag, so `-out output.mkv` carries a switched-off English (and
Chinese) track alongside the burned-in captions.

If you would rather toggle captions in your player than have them burned in, use
`--subtitles soft` — that embeds an enabled track and leaves the picture
untouched, which also lets the video be stream-copied instead of re-encoded.

---

## Requirements

- Linux or macOS, **ffmpeg** on `PATH`
- **Python 3.9–3.11** (3.11 recommended — every dependency ships a wheel for it)
- ~8 GB free disk for models
- Optional: an NVIDIA GPU. 4 GB VRAM is enough; without a GPU everything still
  runs, roughly 5–8× slower.

`setup.sh` installs no system packages, so it needs no `sudo`. If ffmpeg is
missing, install it first:

```bash
sudo apt install ffmpeg        # Debian/Ubuntu
brew install ffmpeg            # macOS
```

---

## Setup

```bash
./setup.sh                          # venv + dependencies + models
./setup.sh --with-demucs            # + background music separation
./setup.sh --with-kokoro            # + the higher quality Kokoro voice
./setup.sh --cpu                    # force the CPU build of PyTorch
./setup.sh --python /usr/bin/python3.11
./setup.sh --asr-model medium --voice en_US-amy-medium
```

It creates `.venv/`, installs everything into it, and downloads all models into
`models/`. Nothing is written outside the project directory, so moving the
project to another machine is a copy plus one `./setup.sh` run.

Re-running `setup.sh` is safe — it reuses the venv and skips models already
downloaded.

---

## Usage

```bash
# local file
./convert_video.sh input.mp4 -out output.mp4

# YouTube (or anything else yt-dlp supports)
./convert_video.sh "https://www.youtube.com/watch?v=XXXX" -out output.mp4

# keep the original music and sound effects under the English voice
./convert_video.sh input.mp4 -out output.mp4 --keep-background

# quick draft to check the translation before committing to a full run
./convert_video.sh input.mp4 -out draft.mp4 --asr-model small --crf 28

# full option list
./convert_video.sh --help
```

### Frequently used options

| Option | Effect |
|---|---|
| `--keep-background` | run Demucs, mix the dub over the original music/SFX |
| `--asr-model small\|medium\|large-v3` | accuracy vs. speed |
| `--voice en_US-ryan-high` | pick another Piper voice (`--list-voices`) |
| `--tts kokoro` | use Kokoro instead of Piper (more natural, slower) |
| `--subtitles burn\|soft\|both\|none` | how captions are attached |
| `--max-speedup 1.6` | allow faster speech when English overruns |
| `--chunk-minutes 20` | audio chunk size for long videos (`0` disables) |
| `--device cpu` | ignore the GPU |
| `--force-from tts` | re-run from a stage, reusing earlier results |
| `--glossary terms.json` | force preferred English for specific terms |

### Long videos

Audio is processed in 20-minute chunks by default, which bounds memory use and
gives the run natural checkpoints. Chunks are **not** cut at a fixed offset: each
boundary is moved to the nearest silence, because a cut landing mid-word makes
Whisper mis-transcribe the words on both sides of it.

Whisper also tends to hallucinate a line right after a chunk starts, usually
echoing speech from elsewhere in the video. Those echoes are detected and
dropped where they overlap a neighbouring line. Keep `--chunk-minutes` at 5 or
above: shorter chunks measurably degrade transcription.

### Resuming

Every stage caches its result in `work/<job-id>/`. If a run is interrupted, or
you change one setting, re-run the same command — completed stages are reused
and reported as `(reused)`. To redo a stage deliberately:

```bash
./convert_video.sh input.mp4 -out output.mp4 --force-from translate
```

Stages, in order: `fetch separate asr segment translate tts subtitles mux`.

---

## How the timing works

English takes roughly 20–40 % longer to say than the Chinese it came from, so a
naive dub drifts further out of sync with every line. This pipeline never moves
a line's start time. For each sentence it computes a budget — the sentence's own
span plus 90 % of the silence that follows — and fits the speech into it:

1. Synthesise at the normal rate. If it fits, done.
2. If not, **re-synthesise faster** via Piper's `length_scale` (down to 0.78).
   Re-rendering keeps the prosody natural; squeezing a waveform does not.
3. If it still overruns, **time-stretch** with `rubberband` (pitch preserved),
   up to a total of `--max-speedup` (1.45 by default).
4. If even that is not enough, the line is allowed to overlap slightly into the
   next one rather than pushing everything later.

The run prints how many lines landed in each category. If more than a quarter
still overrun, raise `--max-speedup` or shorten the translation.

Captions are timed to the **English** audio, not the Chinese source, so the text
on screen matches what is being said.

---

## Configuration

`config/default.yaml` holds every tunable with comments. Override per run:

```bash
./convert_video.sh input.mp4 -out output.mp4 --config my.yaml
./convert_video.sh --print-config          # show the effective settings
```

`my.yaml` only needs the keys you want to change:

```yaml
tts:
  voice: en_US-ryan-high
sync:
  max_speedup: 1.6
subtitles:
  font_size: 26
```

### Glossary

Machine translation is inconsistent with names and jargon. A glossary file fixes
the English side after translation:

```json
{ "Hongmeng": "HarmonyOS", "large model": "LLM" }
```

```bash
./convert_video.sh input.mp4 -out output.mp4 --glossary terms.json
```

---

## Performance

Measured end to end on the machine this was built on — GTX 1650 (4 GB) and an
i7-10750H — for a 93-second 1080p video:

| Stage | Time |
|---|---|
| Download + audio extract | 40 s (network bound) |
| Demucs separation *(optional)* | 12 s |
| Transcription (`large-v3`, GPU) | 25 s |
| Translation (NLLB-600M, GPU) | 16 s |
| Speech synthesis (Piper, CPU) | 8 s |
| Mux + burn-in (NVENC, 1080p) | 12 s |
| **Total** (no separation, excluding download) | **1 min** |

Each model-loading stage carries 10–15 s of fixed overhead, so longer videos
run proportionally faster than this table suggests. Steady-state throughput on
this GPU is roughly 4× realtime for transcription and 8× for both separation
and encoding. On CPU only, budget 5–8× longer and use `--asr-model small`.

**Low on VRAM?** The stages run strictly one after another and free their model
before the next starts, so only one model is resident at a time. `large-v3`
loads as `int8_float16` (~1.6 GB) whenever the GPU has less than 6 GB.

**More VRAM than 4 GB?** The single biggest quality win is a larger translation
model:

```bash
./convert_video.sh input.mp4 -out output.mp4 \
    --translate-model facebook/nllb-200-distilled-1.3B
```

## Troubleshooting

**`Could not load library libcudnn_ops.so.9`**
CTranslate2 cannot find the CUDA libraries. `convert_video.sh` sets
`LD_LIBRARY_PATH` for you — run through that script rather than calling
`python -m vtrans` directly.

**CUDA out of memory**
Another process is using the GPU, or the model is too large. Try
`--compute-type int8`, `--asr-model medium`, or `--device cpu`.

**No speech was recognised**
The audio may not be Chinese (`--language yue` for Cantonese, `--language en`
for English), or VAD dropped everything: try `--no-vad`.

**Captions are too small / too large**
`--font-size 26`. The subtitle style is resolution-independent, so one value
works for 720p and 1080p alike.

**The dub sounds rushed**
Too many lines are being compressed. Use `--max-speedup 1.25` to cap it, accept
more overlap, or switch to a more concise translation with `--translator opus`.

**A voice you want is missing**
Any voice from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
works: drop its `.onnx` and `.onnx.json` into `models/piper/` and pass
`--voice <name>`.

---

## Possible improvements

Deliberately not implemented here, in rough order of value:

1. **LLM translation.** NLLB is sentence-level and has no memory of context. A
   small instruct model (Qwen2.5-7B, Gemma-3) translating with the previous few
   lines as context handles pronouns, names and register far better. Needs more
   VRAM than 4 GB for a useful size, so it is an upgrade, not a default.
2. **Speaker diarisation + per-speaker voices.** `pyannote.audio` labels who is
   speaking; mapping each speaker to a different Piper voice makes multi-person
   content dramatically clearer.
3. **Voice cloning.** XTTS-v2 or F5-TTS can keep the original speaker's timbre
   in English. Both are heavier, and XTTS-v2's licence is non-commercial.
4. **Translating for length.** Ask the translator for a shorter rendering when a
   line would overrun. This attacks the timing problem at the source instead of
   compressing audio afterwards.
5. **Forced alignment.** Running the English audio back through an aligner
   (WhisperX / MFA) would tighten caption timings to the word.

---

## Layout

```
setup.sh              install everything into ./.venv and ./models
convert_video.sh      entry point (sets CUDA paths, then runs the CLI)
config/default.yaml   every tunable, commented
requirements.txt      pinned core dependencies
src/vtrans/
  cli.py              argument parsing
  pipeline.py         stage orchestration, caching and resume
  media.py            yt-dlp, audio extraction, silence-aware chunking
  asr.py              faster-whisper transcription
  segment.py          ASR output -> sentences
  translate.py        NLLB / opus-mt
  tts.py              Piper / Kokoro engines
  sync.py             timing fit and dub track assembly
  separate.py         Demucs background separation
  subtitles.py        SRT and ASS generation
  mux.py              final ffmpeg assembly
  download.py         model pre-fetch
models/               downloaded models (git-ignored)
work/                 per-job intermediates (git-ignored)
```

## Licences

Code here is yours to use. The models have their own terms: Whisper (MIT),
NLLB-200 (CC-BY-NC 4.0 — **non-commercial**), Piper voices (MIT/CC, per voice),
Kokoro (Apache-2.0), Demucs (MIT). Check NLLB's licence in particular before
commercial use. `--translator opus` uses Helsinki-NLP's OPUS-MT models, which
are more permissively licensed — check the specific model card.
