"""Audio extraction utilities using ffmpeg."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from whisper_srt.config import (
    CHUNK_OVERLAP,
    DEFAULT_CHUNK_DURATION,
    LONG_VIDEO_THRESHOLD,
    SUPPORTED_EXTENSIONS,
)


@dataclass(frozen=True)
class AudioChunk:
    path: Path  # path to the WAV temp file
    offset: float  # start time in original audio (seconds) — 0.0 for no chunking


def check_ffmpeg() -> None:
    """Raise RuntimeError with install instructions if ffmpeg is not on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install it via: brew install ffmpeg (macOS) or "
            "apt install ffmpeg (Linux)."
        ) from exc


def get_audio_duration(video_path: Path) -> float:
    """Return duration in seconds using ffprobe. Raise ValueError if it fails."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"ffprobe failed to read duration from {video_path}: {exc.stderr}"
        ) from exc


def extract_audio(video_path: Path) -> Path:
    """
    Extract full audio as 16kHz mono WAV to a temp file.

    Returns Path to the temp WAV file (caller must clean up).

    Raises:
      FileNotFoundError if video_path does not exist
      ValueError if extension not in SUPPORTED_EXTENSIONS
      RuntimeError if ffmpeg is not installed or extraction fails
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{video_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    check_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                "-y",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg extraction failed: {exc.stderr}"
        ) from exc

    return output_path


def extract_audio_chunks(video_path: Path) -> list[AudioChunk]:
    """
    For videos longer than LONG_VIDEO_THRESHOLD, split into overlapping chunks.

    Each chunk is DEFAULT_CHUNK_DURATION seconds with CHUNK_OVERLAP overlap.
    For short videos, returns a single AudioChunk with offset=0.0.
    Uses ffmpeg -ss and -t flags for seeking.

    Returns list of AudioChunk (caller must clean up paths).
    """
    duration = get_audio_duration(video_path)

    if duration <= LONG_VIDEO_THRESHOLD:
        wav_path = extract_audio(video_path)
        return [AudioChunk(path=wav_path, offset=0.0)]

    chunks: list[AudioChunk] = []
    start = 0.0

    while start < duration:
        chunk_duration = min(DEFAULT_CHUNK_DURATION, duration - start)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    str(start),
                    "-i",
                    str(video_path),
                    "-t",
                    str(chunk_duration),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    "-y",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg chunk extraction failed at offset {start}s: {exc.stderr}"
            ) from exc

        chunks.append(AudioChunk(path=output_path, offset=start))

        next_start = start + DEFAULT_CHUNK_DURATION - CHUNK_OVERLAP
        if next_start >= duration:
            break
        start = next_start

    return chunks
