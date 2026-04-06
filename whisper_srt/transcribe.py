"""Transcription utilities using mlx-whisper."""

from pathlib import Path

from tqdm import tqdm

from whisper_srt import config
from whisper_srt.audio import AudioChunk
from whisper_srt.config import CHUNK_OVERLAP
from whisper_srt.srt import Segment

try:
    import mlx_whisper as _mlx_whisper
except ImportError:
    _mlx_whisper = None  # type: ignore[assignment]


def transcribe(
    audio_path: Path,
    *,
    language: str | None = None,
    model: str = config.MODEL_NAME,
) -> list[Segment]:
    """
    Transcribe audio file using mlx_whisper.
    Returns list of Segment (immutable, typed).
    Downloads model on first run (~3 GB, cached to ~/.cache/huggingface/).
    """
    if _mlx_whisper is None:
        raise RuntimeError(
            "mlx-whisper is not installed. Install it via: pip install mlx-whisper"
        )

    result = _mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        verbose=False,
    )

    try:
        raw_segments = result["segments"]
    except KeyError:
        return []

    return [
        Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
        )
        for seg in raw_segments
    ]


def transcribe_chunks(
    chunks: list[AudioChunk],
    *,
    language: str | None = None,
    model: str = config.MODEL_NAME,
    quiet: bool = False,
) -> list[Segment]:
    """
    Transcribe a list of AudioChunks (from extract_audio_chunks).
    Merges segments from each chunk by adding chunk.offset to all timestamps.
    Deduplicates segments in the overlap region: drops segments from later chunks
    whose adjusted start time falls within the overlap period of the previous chunk's end.
    Returns a merged, time-ordered list of Segments.
    """
    merged: list[Segment] = []

    chunk_iter = (
        tqdm(chunks, desc="Transcribing chunks", unit="chunk", disable=quiet)
        if len(chunks) > 1
        else chunks
    )
    for i, chunk in enumerate(chunk_iter):
        raw_segments = transcribe(chunk.path, language=language, model=model)

        for seg in raw_segments:
            adjusted_start = seg.start + chunk.offset
            adjusted_end = seg.end + chunk.offset

            if i > 0:
                # previous_chunk_end is the offset of the current chunk
                # (chunks are contiguous with overlap)
                previous_chunk_end = chunk.offset
                overlap_start = previous_chunk_end - CHUNK_OVERLAP
                if adjusted_start < overlap_start:
                    continue

            merged.append(Segment(start=adjusted_start, end=adjusted_end, text=seg.text))

    return sorted(merged, key=lambda s: s.start)
