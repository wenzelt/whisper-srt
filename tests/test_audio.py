"""Unit tests for whisper_srt.audio module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_srt.audio import (
    check_ffmpeg,
    extract_audio,
    extract_audio_chunks,
    get_audio_duration,
)
from whisper_srt.config import LONG_VIDEO_THRESHOLD

# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------


def test_check_ffmpeg_missing() -> None:
    """Patching subprocess.run to raise FileNotFoundError triggers RuntimeError."""
    with (
        patch("whisper_srt.audio.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match="ffmpeg is not installed"),
    ):
        check_ffmpeg()


def test_check_ffmpeg_present() -> None:
    """When subprocess.run succeeds, check_ffmpeg raises nothing."""
    mock_result = MagicMock()
    with patch("whisper_srt.audio.subprocess.run", return_value=mock_result):
        check_ffmpeg()  # should not raise


# ---------------------------------------------------------------------------
# extract_audio
# ---------------------------------------------------------------------------


def test_extract_audio_file_not_found(tmp_path: Path) -> None:
    """Passing a non-existent path raises FileNotFoundError."""
    missing = tmp_path / "ghost.mp4"
    with pytest.raises(FileNotFoundError):
        extract_audio(missing)


def test_extract_audio_unsupported_extension(tmp_path: Path) -> None:
    """Passing a .txt file raises ValueError about unsupported extension."""
    txt_file = tmp_path / "transcript.txt"
    txt_file.touch()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        extract_audio(txt_file)


def test_extract_audio_creates_wav(tmp_path: Path) -> None:
    """When subprocess.run succeeds, returned path ends in .wav."""
    video = tmp_path / "clip.mp4"
    video.touch()

    mock_result = MagicMock()
    with patch("whisper_srt.audio.subprocess.run", return_value=mock_result):
        result = extract_audio(video)

    assert result.suffix == ".wav"
    # Clean up the temp file that was created
    result.unlink(missing_ok=True)


def test_extract_audio_ffmpeg_failure(tmp_path: Path) -> None:
    """When ffmpeg raises CalledProcessError, RuntimeError contains stderr."""
    video = tmp_path / "clip.mp4"
    video.touch()

    stderr_text = "Error opening input file"
    error = subprocess.CalledProcessError(1, "ffmpeg", stderr=stderr_text)

    call_count = 0

    def side_effect(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        # First call is check_ffmpeg's version check — let it succeed
        if call_count == 1:
            return MagicMock()
        raise error

    with (
        patch("whisper_srt.audio.subprocess.run", side_effect=side_effect),
        pytest.raises(RuntimeError, match=stderr_text),
    ):
        extract_audio(video)


# ---------------------------------------------------------------------------
# extract_audio_chunks
# ---------------------------------------------------------------------------


def test_extract_audio_chunks_short_video(tmp_path: Path) -> None:
    """For a short video (60s), a single AudioChunk with offset=0.0 is returned."""
    video = tmp_path / "short.mp4"
    video.touch()

    mock_result = MagicMock()

    with (
        patch(
            "whisper_srt.audio.get_audio_duration",
            return_value=60.0,
        ),
        patch("whisper_srt.audio.subprocess.run", return_value=mock_result),
    ):
        chunks = extract_audio_chunks(video)

    assert len(chunks) == 1
    assert chunks[0].offset == 0.0
    assert chunks[0].path.suffix == ".wav"
    chunks[0].path.unlink(missing_ok=True)


def test_extract_audio_chunks_long_video(tmp_path: Path) -> None:
    """For a long video (7500s > 7200), multiple chunks with correct offsets are returned."""
    video = tmp_path / "long.mp4"
    video.touch()

    mock_result = MagicMock()

    with (
        patch(
            "whisper_srt.audio.get_audio_duration",
            return_value=float(LONG_VIDEO_THRESHOLD + 300),  # 7500s
        ),
        patch("whisper_srt.audio.subprocess.run", return_value=mock_result),
    ):
        chunks = extract_audio_chunks(video)

    assert len(chunks) > 1

    # First chunk must start at 0
    assert chunks[0].offset == 0.0

    # Each subsequent chunk offset must be strictly greater than the previous
    for i in range(1, len(chunks)):
        assert chunks[i].offset > chunks[i - 1].offset

    # All paths must end in .wav
    for chunk in chunks:
        assert chunk.path.suffix == ".wav"
        chunk.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# get_audio_duration
# ---------------------------------------------------------------------------


def test_get_audio_duration_success(tmp_path: Path) -> None:
    """ffprobe returning '123.456\\n' is parsed to 123.456."""
    video = tmp_path / "clip.mp4"
    video.touch()

    mock_result = MagicMock()
    mock_result.stdout = "123.456\n"

    with patch("whisper_srt.audio.subprocess.run", return_value=mock_result):
        duration = get_audio_duration(video)

    assert duration == pytest.approx(123.456)


def test_get_audio_duration_failure(tmp_path: Path) -> None:
    """CalledProcessError from ffprobe is re-raised as ValueError."""
    video = tmp_path / "clip.mp4"
    video.touch()

    error = subprocess.CalledProcessError(1, "ffprobe", stderr="no such file")

    with (
        patch("whisper_srt.audio.subprocess.run", side_effect=error),
        pytest.raises(ValueError, match="ffprobe failed"),
    ):
        get_audio_duration(video)
