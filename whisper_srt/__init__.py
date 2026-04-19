"""whisper-srt: Local SRT subtitle generator for Apple M1."""

from whisper_srt.audio import AudioChunk, extract_audio, extract_audio_chunks
from whisper_srt.srt import Segment, segments_to_srt, write_srt
from whisper_srt.transcribe import transcribe, transcribe_chunks

__version__ = "0.1.0"

__all__ = [
    "AudioChunk",
    "extract_audio",
    "extract_audio_chunks",
    "Segment",
    "segments_to_srt",
    "write_srt",
    "transcribe",
    "transcribe_chunks",
]
