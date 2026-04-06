# whisper-srt

Local SRT subtitle generator for Apple Silicon. Extracts audio from video files and transcribes them using mlx-whisper — everything runs on-device, no cloud API required.

## Prerequisites

- **Python 3.10+**
- **ffmpeg** — `brew install ffmpeg`
- **uv** — `pip install uv`

## Installation

```bash
git clone https://github.com/your-username/whisper-srt.git
cd whisper-srt
uv sync
```

> **First run note:** Downloads the whisper-large-v3 model (~3 GB) on first use, cached to `~/.cache/huggingface/`. Subsequent runs use the cache.

## Usage

### Single file

```bash
python -m whisper_srt movie.mov
```

### Batch processing

```bash
python -m whisper_srt *.mp4
```

### Force language

```bash
python -m whisper_srt -l en movie.mov
```

### Custom output directory

```bash
python -m whisper_srt -o ./subs/ movie.mov
```

### Smaller model (faster, less accurate)

```bash
python -m whisper_srt -m mlx-community/whisper-medium-mlx movie.mov
```

### All options

```
usage: whisper-srt [-h] [-l LANGUAGE] [-o OUTPUT_DIR] [-m MODEL] video_files [video_files ...]

positional arguments:
  video_files           One or more video file paths.

options:
  -l, --language        Force language code (e.g., 'en', 'de'). Auto-detected if omitted.
  -o, --output-dir      Output directory. Defaults to same directory as the video.
  -m, --model           Whisper model (HuggingFace repo or local path).
```

## Supported formats

| Extension | Container         |
|-----------|-------------------|
| `.mp4`    | MPEG-4            |
| `.mov`    | QuickTime         |
| `.mkv`    | Matroska          |
| `.avi`    | AVI               |
| `.webm`   | WebM              |
| `.m4v`    | iTunes Video      |
| `.mts`    | AVCHD             |
| `.m2ts`   | Blu-ray AVCHD     |

## Performance estimates

Tested on Apple M1 with whisper-large-v3:

| Video length | Approximate transcription time |
|--------------|-------------------------------|
| 5 minutes    | 1–2 minutes                   |
| 30 minutes   | 8–12 minutes                  |
| 2 hours      | 30–50 minutes                 |

Videos longer than 2 hours are automatically split into 30-minute chunks with overlap to avoid memory pressure.

## How it works

1. **ffmpeg** extracts the audio track as a 16 kHz mono WAV file.
2. **mlx-whisper** transcribes the audio using Apple's MLX framework (runs on the Neural Engine / GPU).
3. Timestamps and text are formatted into standard SRT output.

## Troubleshooting

### ffmpeg not found

```
RuntimeError: ffmpeg is not installed or not on PATH.
```

Install via Homebrew: `brew install ffmpeg`

### Model download fails

If the first run fails during model download, check your internet connection and disk space (~3 GB free required). The model is cached at `~/.cache/huggingface/` after the first successful download.

### Unsupported file extension

whisper-srt only processes known video containers. Convert unsupported formats to `.mp4` first:

```bash
ffmpeg -i input.flv output.mp4
```
