"""CLI entry point for whisper-srt."""

import argparse
import sys
from pathlib import Path

from whisper_srt import config
from whisper_srt.audio import (
    check_ffmpeg,
    extract_audio,
    extract_audio_chunks,
    get_audio_duration,
)
from whisper_srt.srt import segments_to_srt, write_srt
from whisper_srt.transcribe import transcribe, transcribe_chunks


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
        type=str,
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

    args = parser.parse_args()

    # Pre-flight: verify ffmpeg is available
    try:
        check_ffmpeg()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    total_count = len(args.video_files)
    success_count = 0

    for video_path in args.video_files:
        temp_files: list[Path] = []
        try:
            print(f"[1/4] Extracting audio from {video_path.name}...")

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

                print(
                    f"[2/4] Transcribing {video_path.name} ({duration:.0f}s) with {args.model}..."
                )
                segments = transcribe_chunks(chunks, language=args.language, model=args.model)
            else:
                audio_path = extract_audio(video_path)
                temp_files = [audio_path]

                print(
                    f"[2/4] Transcribing {video_path.name} ({duration:.0f}s) with {args.model}..."
                )
                segments = transcribe(audio_path, language=args.language, model=args.model)

            print(f"[3/4] Formatting SRT for {video_path.name}...")
            srt_content = segments_to_srt(segments)

            print(f"[4/4] Writing {output_path}...")
            write_srt(srt_content, output_path)

            success_count += 1
            print(f"✓ Done: {output_path} ({len(segments)} segments)")

        except Exception as e:  # noqa: BLE001
            print(f"✗ Error processing {video_path.name}: {e}")

        finally:
            for tmp in temp_files:
                tmp.unlink(missing_ok=True)

    print(f"\nProcessed {success_count}/{total_count} files.")
