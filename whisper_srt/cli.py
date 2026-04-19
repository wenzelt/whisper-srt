"""CLI entry point for whisper-srt."""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from whisper_srt import config
from whisper_srt.audio import (
    check_ffmpeg,
    extract_audio,
    extract_audio_chunks,
    get_audio_duration,
)
from whisper_srt.srt import segments_to_srt, write_srt
from whisper_srt.transcribe import transcribe, transcribe_chunks

console = Console()
logger = logging.getLogger("whisper-srt")

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


def _validate_language(value: str) -> str:
    if not _LANGUAGE_RE.match(value):
        raise argparse.ArgumentTypeError(
            f"Invalid language code {value!r}. Expected 2-3 lowercase letters (e.g. 'en', 'de')."
        )
    return value


def _setup_logging(quiet: bool) -> None:
    """Configure logging based on quiet flag."""
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
        force=True,
    )


def _process_single_video(
    video_path: Path, args: argparse.Namespace, progress: Optional[Progress] = None
) -> bool:
    """Process a single video file. Returns True if successful, False otherwise."""
    temp_files: list[Path] = []
    try:
        # Determine output path
        if args.output_dir is not None:
            output_path = args.output_dir / (video_path.stem + ".srt")
        else:
            output_path = video_path.with_suffix(".srt")

        if not args.overwrite and output_path.exists():
            logger.info("⏭ [yellow]Skipping[/yellow] %s (SRT already exists)", video_path.name)
            return True

        # Check duration to decide chunking strategy
        duration = get_audio_duration(video_path)

        if duration > config.LONG_VIDEO_THRESHOLD:
            with console.status(f"[bold blue]Extracting audio chunks[/bold blue] from {video_path.name}..."):
                chunks = extract_audio_chunks(video_path)
                temp_files = [chunk.path for chunk in chunks]

            logger.info(
                "Transcribing %s (%.0fs) in [cyan]%d chunks[/cyan] using [green]%s[/green]...",
                video_path.name,
                duration,
                len(chunks),
                args.model,
            )
            
            # Use progress for chunks if we have multiple
            if progress:
                chunk_task = progress.add_task(
                    f"Transcribing {video_path.name}", total=len(chunks)
                )
                segments = transcribe_chunks(
                    chunks, 
                    language=args.language, 
                    model=args.model, 
                    quiet=args.quiet,
                    progress_callback=lambda: progress.advance(chunk_task)
                )
                progress.remove_task(chunk_task)
            else:
                segments = transcribe_chunks(
                    chunks, language=args.language, model=args.model, quiet=args.quiet
                )
        else:
            with console.status(f"[bold blue]Extracting audio[/bold blue] from {video_path.name}..."):
                audio_path = extract_audio(video_path)
                temp_files = [audio_path]

            with console.status(
                f"[bold green]Transcribing[/bold green] {video_path.name} (%.0fs) with {args.model}..."
            ):
                segments = transcribe(audio_path, language=args.language, model=args.model)

        with console.status(f"[bold cyan]Formatting SRT[/bold cyan] for {video_path.name}..."):
            srt_content = segments_to_srt(segments)

        with console.status(f"[bold magenta]Writing[/bold magenta] {output_path}..."):
            write_srt(srt_content, output_path)

        logger.info("✓ [green]Done:[/green] %s (%d segments)", output_path, len(segments))
        return True

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("✗ [red]Error processing[/red] %s: %s", video_path.name, e)
        return False

    finally:
        for tmp in temp_files:
            tmp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing SRT files. By default, existing files are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.quiet)

    # Pre-flight: verify ffmpeg is available
    try:
        check_ffmpeg()
    except RuntimeError as e:
        logger.error("[red]%s[/red]", e)
        sys.exit(1)

    total_count = len(args.video_files)
    success_count = 0

    if args.quiet:
        for video_path in args.video_files:
            if _process_single_video(video_path, args):
                success_count += 1
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # We use two progress bars if there are multiple files
            if total_count > 1:
                overall_task = progress.add_task("Overall Progress", total=total_count)
                for video_path in args.video_files:
                    if _process_single_video(video_path, args, progress):
                        success_count += 1
                    progress.advance(overall_task)
            else:
                # For a single file, just pass the progress object to handle chunk progress
                if _process_single_video(args.video_files[0], args, progress):
                    success_count += 1

    logger.info("\n[bold]Processed %d/%d files.[/bold]", success_count, total_count)
