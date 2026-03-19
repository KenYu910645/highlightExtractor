"""Audio utilities for offline speaker labeling and enrollment prep."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


DEFAULT_FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
DEFAULT_FFPROBE = Path(r"C:\ffmpeg\bin\ffprobe.exe")

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
}

SUPPORTED_SOURCE_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}


def is_supported_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def is_supported_source_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS


def resolve_media_binary(binary_name: str, override: str | None = None) -> str:
    """Resolve an ffmpeg/ffprobe executable from override, PATH, or default Windows install."""
    if override:
        return override

    env_var = "FFMPEG_BIN" if binary_name == "ffmpeg" else "FFPROBE_BIN"
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env

    from_path = shutil.which(binary_name)
    if from_path:
        return from_path

    default_path = DEFAULT_FFMPEG if binary_name == "ffmpeg" else DEFAULT_FFPROBE
    if default_path.exists():
        return str(default_path)

    return binary_name


def decode_audio_to_array(path: str, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """
    Decode audio with ffmpeg into mono float32 samples in [-1, 1].
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                resolve_media_binary("ffmpeg"),
                "-y",
                "-i",
                path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        return load_wav_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_wav_file(path: str) -> tuple[np.ndarray, int]:
    """
    Load a mono/stereo WAV file as mono float32 samples in [-1, 1].
    """
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def write_wav_file(path: str | Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Write mono float32 samples in [-1, 1] to a 16-bit PCM WAV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def slice_audio(samples: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    """
    Slice audio safely to the requested time range.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if end_sec <= start_sec or len(samples) == 0:
        return np.zeros(0, dtype=np.float32)

    start_idx = max(0, int(math.floor(start_sec * sample_rate)))
    end_idx = min(len(samples), int(math.ceil(end_sec * sample_rate)))

    if end_idx <= start_idx:
        return np.zeros(0, dtype=np.float32)
    return samples[start_idx:end_idx].astype(np.float32, copy=False)


def estimate_voiced_seconds(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    hop_ms: int = 15,
    energy_threshold: float = 0.015,
) -> float:
    """
    Lightweight energy-based speech estimate used to skip near-silent segments.
    """
    if len(samples) == 0 or sample_rate <= 0:
        return 0.0

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))

    voiced_frames = 0
    total_frames = 0

    for start in range(0, max(1, len(samples) - frame_len + 1), hop_len):
        frame = samples[start:start + frame_len]
        if len(frame) < frame_len:
            break
        total_frames += 1
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
        if rms >= energy_threshold and zcr <= 0.4:
            voiced_frames += 1

    if total_frames == 0:
        return 0.0
    return voiced_frames * (hop_len / sample_rate)
