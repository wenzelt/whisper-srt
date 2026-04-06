from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float    # seconds
    text: str     # subtitle text (stripped)


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format: HH:MM:SS,mmm"""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    """
    Convert a list of Segments to a complete SRT string.
    SRT format:
      1
      00:00:01,000 --> 00:00:04,500
      Hello world

      2
      ...
    Empty segments list returns empty string "".
    """
    if not segments:
        return ""

    blocks = []
    for i, segment in enumerate(segments, start=1):
        start_ts = format_timestamp(segment.start)
        end_ts = format_timestamp(segment.end)
        text = segment.text.strip()
        blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")

    return "\n".join(blocks).rstrip()


def write_srt(srt_content: str, output_path: Path) -> Path:
    """
    Write SRT content to output_path with UTF-8-BOM encoding.
    Creates parent directories if needed.
    Returns output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt_content, encoding="utf-8-sig")
    return output_path
