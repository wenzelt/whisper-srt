from pathlib import Path

from whisper_srt.srt import Segment, format_timestamp, segments_to_srt, write_srt

# --- format_timestamp ---

def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00,000"


def test_format_timestamp_milliseconds():
    assert format_timestamp(1.5) == "00:00:01,500"


def test_format_timestamp_minutes():
    assert format_timestamp(61.0) == "00:01:01,000"


def test_format_timestamp_hours():
    assert format_timestamp(3661.123) == "01:01:01,123"


def test_format_timestamp_near_full_day():
    assert format_timestamp(86399.999) == "23:59:59,999"


# --- segments_to_srt ---

def test_segments_to_srt_empty():
    assert segments_to_srt([]) == ""


def test_segments_to_srt_single():
    segments = [Segment(start=1.0, end=4.5, text="Hello world")]
    result = segments_to_srt(segments)
    assert result == "1\n00:00:01,000 --> 00:00:04,500\nHello world"


def test_segments_to_srt_multiple():
    segments = [
        Segment(start=1.0, end=4.5, text="Hello world"),
        Segment(start=5.0, end=8.0, text="Goodbye"),
    ]
    result = segments_to_srt(segments)
    expected = (
        "1\n00:00:01,000 --> 00:00:04,500\nHello world\n"
        "\n"
        "2\n00:00:05,000 --> 00:00:08,000\nGoodbye"
    )
    assert result == expected


def test_segments_to_srt_unicode():
    segments = [Segment(start=0.0, end=2.0, text="こんにちは")]
    result = segments_to_srt(segments)
    assert "こんにちは" in result


# --- write_srt ---

def test_write_srt_creates_file(tmp_path: Path) -> None:
    output = tmp_path / "output.srt"
    content = "1\n00:00:00,000 --> 00:00:01,000\nHello"
    write_srt(content, output)
    assert output.exists()
    assert output.read_text(encoding="utf-8-sig") == content


def test_write_srt_utf8_bom(tmp_path: Path) -> None:
    output = tmp_path / "output.srt"
    write_srt("Hello", output)
    raw = output.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"


def test_write_srt_returns_path(tmp_path: Path) -> None:
    output = tmp_path / "output.srt"
    result = write_srt("Hello", output)
    assert result == output
