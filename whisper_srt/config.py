"""Constants for whisper-srt. All hardcoded values live here."""

MODEL_NAME = "mlx-community/whisper-large-v3-mlx"
SAMPLE_RATE = 16_000  # Hz, required by Whisper
SUPPORTED_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts"}
)
DEFAULT_CHUNK_DURATION = 300  # 5 minutes in seconds
CHUNK_OVERLAP = 30  # seconds of overlap between chunks
LONG_VIDEO_THRESHOLD = 300  # 5 minutes in seconds; videos longer than this get chunked
