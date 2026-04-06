"""CLI entry point for whisper-srt."""

import argparse
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

from whisper_srt import config
from whisper_srt.audio import (
    check_ffmpeg,
    extract_audio,
    extract_audio_chunks,
    get_audio_duration,
)
from whisper_srt.srt import segments_to_srt, write_srt
from whisper_srt.transcribe import transcribe, transcribe_chunks

logger = logging.getLogger(__name__)

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


def _validate_language(value: str) -> str:
    if not _LANGUAGE_RE.match(value):
        raise argparse.ArgumentTypeError(
            f"Invalid language code {value!r}. Expected 2-3 lowercase letters (e.g. 'en', 'de')."
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="whisper-srt",
        description="Generate SRT subtitles from video files using mlx-whisper.",
    )
    parser.add_argument(
        "video_files",
        nargs="+",
        type=Path,
        help="One or more video file paths.",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=_validate_language,
        default=None,
        help="Force language code (e.g., 'en', 'de').",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory; if omitted, SRT goes next to the video.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=config.MODEL_NAME,
        help="Override the Whisper model (HuggingFace repo or local path).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress progress bars and informational messages.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Pre-flight: verify ffmpeg is available
    try:
        check_ffmpeg()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    total_count = len(args.video_files)
    success_count = 0

    file_iter = (
        tqdm(args.video_files, desc="Processing files", unit="file", disable=args.quiet)
        if len(args.video_files) > 1
        else args.video_files
    )
    for video_path in file_iter:
        temp_files: list[Path] = []
        try:
            logger.info("[1/4] Extracting audio from %s...", video_path.name)

            # Determine output path
            if args.output_dir is not None:
                output_path = args.output_dir / (video_path.stem + ".srt")
            else:
                output_path = video_path.with_suffix(".srt")

            # Check duration to decide chunking strategy
            duration = get_audio_duration(video_path)

            if duration > config.LONG_VIDEO_THRESHOLD:
                chunks = extract_audio_chunks(video_path)
                temp_files = [chunk.path for chunk in chunks]

                logger.info(
                    "[2/4] Transcribing %s (%.0fs) with %s...",
                    video_path.name,
                    duration,
                    args.model,
                )
                segments = transcribe_chunks(chunks, language=args.language, model=args.model, quiet=args.quiet)
            else:
                audio_path = extract_audio(video_path)
                temp_files = [audio_path]

                logger.info(
                    "[2/4] Transcribing %s (%.0fs) with %s...",
                    video_path.name,
                    duration,
                    args.model,
                )
                segments = transcribe(audio_path, language=args.language, model=args.model)

            logger.info("[3/4] Formatting SRT for %s...", video_path.name)
            srt_content = segments_to_srt(segments)

            logger.info("[4/4] Writing %s...", output_path)
            write_srt(srt_content, output_path)

            success_count += 1
            logger.info("✓ Done: %s (%d segments)", output_path, len(segments))

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error("✗ Error processing %s: %s", video_path.name, e)

        finally:
            for tmp in temp_files:
                tmp.unlink(missing_ok=True)

    logger.info("\nProcessed %d/%d files.", success_count, total_count)
