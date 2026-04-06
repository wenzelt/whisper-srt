"""Unit tests for whisper_srt.cli module."""

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_srt.srt import Segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_CHECK_FFMPEG = "whisper_srt.cli.check_ffmpeg"
PATCH_GET_DURATION = "whisper_srt.cli.get_audio_duration"
PATCH_EXTRACT_AUDIO = "whisper_srt.cli.extract_audio"
PATCH_EXTRACT_CHUNKS = "whisper_srt.cli.extract_audio_chunks"
PATCH_TRANSCRIBE = "whisper_srt.cli.transcribe"
PATCH_TRANSCRIBE_CHUNKS = "whisper_srt.cli.transcribe_chunks"
PATCH_SEGMENTS_TO_SRT = "whisper_srt.cli.segments_to_srt"
PATCH_WRITE_SRT = "whisper_srt.cli.write_srt"

_DEFAULT_SEGMENT = Segment(0.0, 1.0, "Hello")
_SHORT_DURATION = 60.0
_LONG_DURATION = 8000.0  # > LONG_VIDEO_THRESHOLD (7200)


def _fake_chunk(path: str = "/tmp/test.wav") -> MagicMock:
    chunk = MagicMock()
    chunk.path = Path(path)
    return chunk


# ---------------------------------------------------------------------------
# test_main_no_args_exits
# ---------------------------------------------------------------------------


def test_main_no_args_exits() -> None:
    """argparse exits with an error when required positional args are missing."""
    from whisper_srt.cli import main

    with patch.object(sys, "argv", ["whisper-srt"]), pytest.raises(SystemExit):
        main()


# ---------------------------------------------------------------------------
# test_main_ffmpeg_check_fails_exits
# ---------------------------------------------------------------------------


def test_main_ffmpeg_check_fails_exits() -> None:
    """If check_ffmpeg raises, main() exits with code 1."""
    from whisper_srt.cli import main

    with (
        patch.object(sys, "argv", ["whisper-srt", "video.mp4"]),
        patch(PATCH_CHECK_FFMPEG, side_effect=RuntimeError("ffmpeg missing")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# test_main_single_file_short_video
# ---------------------------------------------------------------------------


def test_main_single_file_short_video(tmp_path: Path) -> None:
    """Short video uses extract_audio + transcribe; SRT is placed next to the video."""
    from whisper_srt.cli import main

    video = tmp_path / "movie.mp4"
    video.touch()
    expected_srt = tmp_path / "movie.srt"

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video)]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_SHORT_DURATION),
        patch(PATCH_EXTRACT_AUDIO, return_value=Path("/tmp/test.wav")) as mock_extract,
        patch(PATCH_TRANSCRIBE, return_value=[_DEFAULT_SEGMENT]) as mock_transcribe,
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n...") as mock_to_srt,
        patch(PATCH_WRITE_SRT, return_value=expected_srt) as mock_write,
        patch.object(Path, "unlink"),
    ):
        main()

    mock_extract.assert_called_once_with(video)
    mock_write.assert_called_once_with("1\n...", expected_srt)
    mock_to_srt.assert_called_once_with([_DEFAULT_SEGMENT])
    mock_transcribe.assert_called_once()


# ---------------------------------------------------------------------------
# test_main_single_file_long_video
# ---------------------------------------------------------------------------


def test_main_single_file_long_video(tmp_path: Path) -> None:
    """Long video (>7200s) uses extract_audio_chunks + transcribe_chunks."""
    from whisper_srt.cli import main

    video = tmp_path / "long.mp4"
    video.touch()

    chunk = _fake_chunk("/tmp/chunk0.wav")

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video)]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_LONG_DURATION),
        patch(PATCH_EXTRACT_CHUNKS, return_value=[chunk]) as mock_chunks,
        patch(PATCH_TRANSCRIBE_CHUNKS, return_value=[_DEFAULT_SEGMENT]) as mock_tc,
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n..."),
        patch(PATCH_WRITE_SRT, return_value=tmp_path / "long.srt"),
        patch.object(Path, "unlink"),
    ):
        main()

    mock_chunks.assert_called_once_with(video)
    mock_tc.assert_called_once()


# ---------------------------------------------------------------------------
# test_main_output_dir
# ---------------------------------------------------------------------------


def test_main_output_dir(tmp_path: Path) -> None:
    """--output-dir places the SRT file inside the specified directory."""
    from whisper_srt.cli import main

    video = tmp_path / "movie.mp4"
    video.touch()
    out_dir = tmp_path / "subs"

    expected_srt = out_dir / "movie.srt"

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video), "-o", str(out_dir)]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_SHORT_DURATION),
        patch(PATCH_EXTRACT_AUDIO, return_value=Path("/tmp/test.wav")),
        patch(PATCH_TRANSCRIBE, return_value=[_DEFAULT_SEGMENT]),
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n..."),
        patch(PATCH_WRITE_SRT, return_value=expected_srt) as mock_write,
        patch.object(Path, "unlink"),
    ):
        main()

    mock_write.assert_called_once_with("1\n...", expected_srt)


# ---------------------------------------------------------------------------
# test_main_language_flag
# ---------------------------------------------------------------------------


def test_main_language_flag(tmp_path: Path) -> None:
    """-l en passes language='en' to transcribe."""
    from whisper_srt.cli import main

    video = tmp_path / "movie.mp4"
    video.touch()

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video), "-l", "en"]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_SHORT_DURATION),
        patch(PATCH_EXTRACT_AUDIO, return_value=Path("/tmp/test.wav")),
        patch(PATCH_TRANSCRIBE, return_value=[_DEFAULT_SEGMENT]) as mock_transcribe,
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n..."),
        patch(PATCH_WRITE_SRT, return_value=tmp_path / "movie.srt"),
        patch.object(Path, "unlink"),
    ):
        main()

    _args, _kwargs = mock_transcribe.call_args
    assert _kwargs.get("language") == "en"


# ---------------------------------------------------------------------------
# test_main_model_flag
# ---------------------------------------------------------------------------


def test_main_model_flag(tmp_path: Path) -> None:
    """-m custom-model passes model='custom-model' to transcribe."""
    from whisper_srt.cli import main

    video = tmp_path / "movie.mp4"
    video.touch()

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video), "-m", "custom-model"]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_SHORT_DURATION),
        patch(PATCH_EXTRACT_AUDIO, return_value=Path("/tmp/test.wav")),
        patch(PATCH_TRANSCRIBE, return_value=[_DEFAULT_SEGMENT]) as mock_transcribe,
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n..."),
        patch(PATCH_WRITE_SRT, return_value=tmp_path / "movie.srt"),
        patch.object(Path, "unlink"),
    ):
        main()

    _args, _kwargs = mock_transcribe.call_args
    assert _kwargs.get("model") == "custom-model"


# ---------------------------------------------------------------------------
# test_main_batch_continues_on_error
# ---------------------------------------------------------------------------


def test_main_batch_continues_on_error(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """When the first file errors, processing continues and second file succeeds."""
    from whisper_srt.cli import main

    video1 = tmp_path / "bad.mp4"
    video1.touch()
    video2 = tmp_path / "good.mp4"
    video2.touch()

    call_count = 0

    def fake_extract(path: Path) -> Path:
        nonlocal call_count
        call_count += 1
        if path.name == "bad.mp4":
            raise RuntimeError("Extraction failed")
        return Path("/tmp/good.wav")

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video1), str(video2)]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_SHORT_DURATION),
        patch(PATCH_EXTRACT_AUDIO, side_effect=fake_extract),
        patch(PATCH_TRANSCRIBE, return_value=[_DEFAULT_SEGMENT]),
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n..."),
        patch(PATCH_WRITE_SRT, return_value=tmp_path / "good.srt"),
        patch.object(Path, "unlink"),
    ):
        main()

    captured = capsys.readouterr()
    assert "Error processing bad.mp4" in captured.out
    assert "Processed 1/2 files." in captured.out


# ---------------------------------------------------------------------------
# test_main_prints_summary
# ---------------------------------------------------------------------------


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """'Processed 1/1 files.' appears in stdout after processing one file."""
    from whisper_srt.cli import main

    video = tmp_path / "movie.mp4"
    video.touch()

    with (
        patch.object(sys, "argv", ["whisper-srt", str(video)]),
        patch(PATCH_CHECK_FFMPEG),
        patch(PATCH_GET_DURATION, return_value=_SHORT_DURATION),
        patch(PATCH_EXTRACT_AUDIO, return_value=Path("/tmp/test.wav")),
        patch(PATCH_TRANSCRIBE, return_value=[_DEFAULT_SEGMENT]),
        patch(PATCH_SEGMENTS_TO_SRT, return_value="1\n..."),
        patch(PATCH_WRITE_SRT, return_value=tmp_path / "movie.srt"),
        patch.object(Path, "unlink"),
    ):
        main()

    captured = capsys.readouterr()
    assert "Processed 1/1 files." in captured.out


# ---------------------------------------------------------------------------
# test___main__
# ---------------------------------------------------------------------------


def test_main_module_invokes_main() -> None:
    """python -m whisper_srt calls cli.main()."""
    with patch("whisper_srt.cli.main") as mock_main:
        runpy.run_module("whisper_srt", run_name="__main__")

    mock_main.assert_called_once()
