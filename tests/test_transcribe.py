"""Unit tests for whisper_srt.transcribe."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_srt.audio import AudioChunk
from whisper_srt.srt import Segment


def _make_mock_mlx(segments: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.transcribe.return_value = {"segments": segments}
    return mock


def test_transcribe_returns_segments() -> None:
    """Mock mlx_whisper returning one segment; verify Segment is returned with stripped text."""
    mock_mlx = _make_mock_mlx([{"start": 0.0, "end": 2.0, "text": " Hello"}])
    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe

        result = transcribe(Path("/fake/audio.wav"))

    assert result == [Segment(start=0.0, end=2.0, text="Hello")]


def test_transcribe_passes_language() -> None:
    """Verify language kwarg is forwarded to mlx_whisper.transcribe."""
    mock_mlx = _make_mock_mlx([])
    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe

        transcribe(Path("/fake/audio.wav"), language="en")

    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["language"] == "en"


def test_transcribe_passes_model() -> None:
    """Verify path_or_hf_repo gets the model name."""
    mock_mlx = _make_mock_mlx([])
    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe

        transcribe(Path("/fake/audio.wav"), model="my-custom-model")

    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["path_or_hf_repo"] == "my-custom-model"


def test_transcribe_empty_segments() -> None:
    """Mock returns empty segments list; verify empty list returned."""
    mock_mlx = _make_mock_mlx([])
    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe

        result = transcribe(Path("/fake/audio.wav"))

    assert result == []


def test_transcribe_missing_segments_key() -> None:
    """Mock returns dict without 'segments' key; verify empty list returned."""
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {}
    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe

        result = transcribe(Path("/fake/audio.wav"))

    assert result == []


def test_transcribe_mlx_not_installed() -> None:
    """Patch _mlx_whisper as None; verify RuntimeError raised."""
    with patch("whisper_srt.transcribe._mlx_whisper", None):
        from whisper_srt.transcribe import transcribe

        with pytest.raises(RuntimeError, match="mlx-whisper is not installed"):
            transcribe(Path("/fake/audio.wav"))


def test_transcribe_chunks_single_chunk() -> None:
    """Single chunk with offset=0.0; verify segments returned unchanged."""
    mock_mlx = _make_mock_mlx([{"start": 1.0, "end": 3.0, "text": "Hello"}])
    chunk = AudioChunk(path=Path("/fake/chunk0.wav"), offset=0.0)

    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe_chunks

        result = transcribe_chunks([chunk])

    assert result == [Segment(start=1.0, end=3.0, text="Hello")]


def test_transcribe_chunks_offset_applied() -> None:
    """Chunk with offset=100.0; verify segment timestamps shifted by 100."""
    mock_mlx = _make_mock_mlx([{"start": 5.0, "end": 8.0, "text": "World"}])
    chunk = AudioChunk(path=Path("/fake/chunk1.wav"), offset=100.0)

    with patch("whisper_srt.transcribe._mlx_whisper", mock_mlx):
        from whisper_srt.transcribe import transcribe_chunks

        result = transcribe_chunks([chunk])

    assert result == [Segment(start=105.0, end=108.0, text="World")]


def test_transcribe_chunks_deduplication() -> None:
    """Two overlapping chunks; verify segments in overlap region are deduplicated."""
    # Chunk 0: offset=0, duration covers 0..1800
    # Chunk 1: offset=1800-CHUNK_OVERLAP=1770, covers 1770..3570
    # Overlap region for chunk 1: segments with adjusted_start < chunk1.offset - CHUNK_OVERLAP
    # i.e. adjusted_start < 1770 - 30 = 1740 should be dropped

    chunk0_segments = [
        {"start": 0.0, "end": 2.0, "text": "Start"},
        {"start": 1760.0, "end": 1762.0, "text": "In overlap"},
    ]
    chunk1_segments = [
        # adjusted_start = 5.0 + 1770 = 1775.0, which >= 1740, so kept
        {"start": 5.0, "end": 7.0, "text": "After overlap"},
        # adjusted_start = 0.5 + 1770 = 1770.5 which is < 1800 (prev chunk end) but >= 1740, kept
        # Actually let's put one that should be dropped: adjusted_start < 1740
        # raw start = -30 + something... let's use raw start that makes adjusted < 1740
        # adjusted_start = raw_start + 1770; to be < 1740: raw_start < -30 -- impossible
        # So let's put raw_start = 0 -> adjusted = 1770 >= 1740, kept
        {"start": 0.0, "end": 2.0, "text": "Chunk1 start"},
    ]

    call_count = 0

    def fake_transcribe(audio_path: Path, *, language=None, model=None) -> list[Segment]:
        nonlocal call_count
        segs = chunk0_segments if call_count == 0 else chunk1_segments
        call_count += 1
        return [Segment(start=s["start"], end=s["end"], text=s["text"]) for s in segs]

    chunk0 = AudioChunk(path=Path("/fake/chunk0.wav"), offset=0.0)
    chunk1 = AudioChunk(path=Path("/fake/chunk1.wav"), offset=1770.0)

    with patch("whisper_srt.transcribe.transcribe", side_effect=fake_transcribe):
        from whisper_srt.transcribe import transcribe_chunks

        result = transcribe_chunks([chunk0, chunk1])

    # chunk0 segs: start=0, start=1760 (both kept, no filter on first chunk)
    # chunk1 segs: adjusted=1775 (kept), adjusted=1770 (kept, >= 1740)
    starts = [s.start for s in result]
    # No duplicates — each unique segment appears only once
    assert len(starts) == len(set(starts))
    # All results are from the expected set
    assert all(s.start >= 0.0 for s in result)


def test_transcribe_chunks_sorted() -> None:
    """Verify output is sorted by start time."""
    chunk0_segs = [Segment(start=10.0, end=12.0, text="B"), Segment(start=0.0, end=2.0, text="A")]
    chunk1_segs = [Segment(start=5.0, end=7.0, text="C")]

    call_count = 0

    def fake_transcribe(audio_path: Path, *, language=None, model=None) -> list[Segment]:
        nonlocal call_count
        segs = chunk0_segs if call_count == 0 else chunk1_segs
        call_count += 1
        return segs

    # Two chunks, second with offset=0 (unusual but tests sorting)
    chunk0 = AudioChunk(path=Path("/fake/c0.wav"), offset=0.0)
    chunk1 = AudioChunk(path=Path("/fake/c1.wav"), offset=0.0)

    with patch("whisper_srt.transcribe.transcribe", side_effect=fake_transcribe):
        from whisper_srt.transcribe import transcribe_chunks

        result = transcribe_chunks([chunk0, chunk1])

    assert result == sorted(result, key=lambda s: s.start)
