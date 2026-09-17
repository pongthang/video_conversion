"""Command line entry point: python -m vtrans <input> --out <output>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .runtime import prepare as prepare_runtime
from .pipeline import STAGES, Pipeline, job_id
from .progress import Cancelled
from .tts import PIPER_VOICES
from .utils import Colors, setup_logging

EPILOG = """\
examples:
  convert a local file
    ./convert_video.sh input.mp4 -out output.mp4

  convert a YouTube video, keeping the original music under the dub
    ./convert_video.sh "https://youtu.be/XXXX" -out output.mp4 --keep-background

  faster, lower quality pass for a quick check
    ./convert_video.sh input.mp4 -out draft.mp4 --asr-model small --subtitles burn

  redo only the speech synthesis of a job you already transcribed
    ./convert_video.sh input.mp4 -out output.mp4 --force-from tts
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_video.sh",
        description="Turn a Chinese-language video into an English-dubbed, "
                    "English-subtitled video using open-source models.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="input video file or URL (YouTube, etc.)")
    # Both -out (as requested) and the conventional --out/-o are accepted.
    p.add_argument("-out", "--out", "-o", dest="out", help="output video path (default: <input>.en.mp4)")
    p.add_argument("--config", help="extra YAML config merged over config/default.yaml")

    g = p.add_argument_group("general")
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], help="compute device")
    g.add_argument("--work-dir", help="directory for intermediate files")
    g.add_argument("--models-dir", help="directory holding downloaded models")
    g.add_argument("--chunk-minutes", type=float,
                   help="split audio into chunks of N minutes before ASR (0 disables)")
    g.add_argument("--force-from", choices=STAGES,
                   help="discard cached artefacts and re-run from this stage")
    g.add_argument("--clean", action="store_true",
                   help="delete intermediate files after a successful run")
    g.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    g.add_argument("--print-config", action="store_true",
                   help="print the effective configuration and exit")
    g.add_argument("--list-voices", action="store_true", help="list bundled Piper voices and exit")

    a = p.add_argument_group("transcription")
    a.add_argument("--asr-model", help="whisper model: tiny|base|small|medium|large-v2|large-v3")
    a.add_argument("--language", help="source language code (default: zh)")
    a.add_argument("--compute-type", help="CTranslate2 compute type (auto|int8|int8_float16|float16)")
    a.add_argument("--no-vad", action="store_true", help="disable the VAD pre-filter")

    t = p.add_argument_group("translation")
    t.add_argument("--translator", choices=["nllb", "opus", "none"], help="translation backend")
    t.add_argument("--translate-model", help="override the translation model id")
    t.add_argument("--glossary", help="JSON file of {source term: preferred English}")

    s = p.add_argument_group("speech synthesis")
    s.add_argument("--tts", choices=["piper", "kokoro"], dest="tts_backend", help="TTS backend")
    s.add_argument("--voice", help=f"piper voice ({', '.join(sorted(PIPER_VOICES))})")
    s.add_argument("--kokoro-voice", help="kokoro voice name (e.g. af_heart, am_michael)")
    s.add_argument("--max-speedup", type=float,
                   help="how much a line may be sped up to fit its slot (default 1.45)")

    b = p.add_argument_group("audio")
    b.add_argument("--keep-background", action="store_true",
                   help="run Demucs and keep the original music/effects under the dub")
    b.add_argument("--background-gain", type=float,
                   help="level in dB applied to the background bed (default -3)")
    b.add_argument("--no-original-audio", action="store_true",
                   help="do not keep the Chinese audio as a second track")

    o = p.add_argument_group("subtitles and output")
    o.add_argument("--subtitles", choices=["burn", "soft", "both", "none"],
                   help="burn captions into the picture, add them as a track, or both")
    o.add_argument("--font-size", type=int, help="subtitle font size (default 22)")
    o.add_argument("--max-line-chars", type=int, help="subtitle wrap width (default 42)")
    o.add_argument("--video-codec", help="libx264 | h264_nvenc | copy | auto")
    o.add_argument("--crf", type=int, help="quality: lower is better (default 20)")
    o.add_argument("--preset", help="x264 preset (default medium)")
    return p


def apply_cli(cfg: Config, args: argparse.Namespace) -> None:
    cfg.apply_overrides({
        "general.device": args.device,
        "general.work_dir": args.work_dir,
        "general.models_dir": args.models_dir,
        "general.keep_work": False if args.clean else None,
        "source.chunk_minutes": args.chunk_minutes,
        "asr.model": args.asr_model,
        "asr.language": args.language,
        "asr.compute_type": args.compute_type,
        "asr.vad_filter": False if args.no_vad else None,
        "translate.backend": args.translator,
        "translate.model": args.translate_model,
        "tts.backend": args.tts_backend,
        "tts.voice": args.voice,
        "tts.kokoro_voice": args.kokoro_voice,
        "sync.max_speedup": args.max_speedup,
        "separate.enabled": True if args.keep_background else None,
        "separate.background_gain_db": args.background_gain,
        "subtitles.mode": args.subtitles,
        "subtitles.font_size": args.font_size,
        "subtitles.max_line_chars": args.max_line_chars,
        "output.video_codec": args.video_codec,
        "output.crf": args.crf,
        "output.preset": args.preset,
        "output.keep_original_audio": False if args.no_original_audio else None,
    })
    # Switching backend without naming a model must not reuse the other one's.
    if args.translator and not args.translate_model:
        from .translate import BACKEND_DEFAULT_MODELS
        cfg.set("translate.model", BACKEND_DEFAULT_MODELS.get(args.translator, ""))


def default_output(input_path: str) -> Path:
    stem = Path(input_path).stem or "output"
    return Path.cwd() / f"{stem}.en.mp4"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_voices:
        print("Piper voices bundled with setup.sh:")
        for name in sorted(PIPER_VOICES):
            print(f"  {name}")
        print("\nAny other voice from https://huggingface.co/rhasspy/piper-voices works:")
        print("  drop its .onnx and .onnx.json into models/piper/ and pass --voice <name>")
        return 0

    cfg = Config.load(args.config)
    apply_cli(cfg, args)

    # Caches, thread limits and the CUDA loader path. convert_video.sh also
    # exports most of these; doing it here too means `python -m vtrans` works
    # on its own and the GUI gets identical treatment without a shell.
    prepare_runtime(cfg.resolve_dir("general.models_dir"))

    if args.print_config:
        print(cfg.dump())
        return 0

    if not args.input:
        parser.error("an input video file or URL is required")

    out_path = Path(args.out).expanduser().resolve() if args.out else default_output(args.input)
    if out_path.suffix.lower() not in (".mp4", ".mkv", ".mov"):
        parser.error(f"output must be .mp4, .mkv or .mov (got '{out_path.suffix}')")

    setup_logging(args.verbose,
                  logfile=cfg.resolve_dir("general.work_dir") / job_id(args.input) / "run.log")

    pipeline = Pipeline(cfg, args.input, out_path,
                        force_from=args.force_from, glossary_path=args.glossary)
    try:
        pipeline.run()
    except (KeyboardInterrupt, Cancelled):
        print("\nInterrupted. Re-run the same command to resume from the last "
              "completed stage.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top level handler
        if args.verbose:
            raise
        print(f"\n{Colors.RED}Error:{Colors.RESET} {exc}\n"
              f"Re-run with -v for the full traceback.", file=sys.stderr)
        return 1

    print(f"\n{Colors.GREEN}Done:{Colors.RESET} {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
