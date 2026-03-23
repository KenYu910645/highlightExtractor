#!/usr/bin/env python3
"""
utils.py — Shared utilities for the highlight extraction pipeline.

Functions shared by preprocess.py (Step 1) and postprocess.py (Step 3).
highlight_extractor.py is kept unchanged and still works standalone.
"""

import os
import re
import subprocess
import sys

import numpy as np

# ── Install check ─────────────────────────────────────────────────────────────
try:
    import whisper
except ImportError:
    print("Installing openai-whisper...")
    subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper",
                    "--break-system-packages", "-q"], check=True)
    import whisper

try:
    import opencc
except ImportError:
    print("Installing opencc-python-reimplemented...")
    subprocess.run([sys.executable, "-m", "pip", "install", "opencc-python-reimplemented",
                    "--break-system-packages", "-q"], check=True)
    import opencc


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING CRITERIA — Traditional Chinese reaction keywords
# ═══════════════════════════════════════════════════════════════════════════════
REACTION_KEYWORDS = {
    # ── Strong exclamations ──────────────────────────────────────────────────
    "哇":    3,
    "哇塞":  5,
    "啊":    2,
    "诶":    2,
    "哎":    2,
    "哎呀":  4,
    "嗯":    1,
    "喔":    1,
    "哦":    1,
    "唉":    2,

    # ── Laughter ─────────────────────────────────────────────────────────────
    "哈哈":  5,
    "嘻嘻":  5,
    "呵呵":  4,
    "嘿嘿":  4,

    # ── Praise / excitement ───────────────────────────────────────────────────
    "好棒":       3,
    "厉害":       3,
    "聪明":       3,
    "棒":         2,
    "好会":       2,
    "好特別":     3,
    "太厲害":     4,
    "你好厲害":   5,
    "好聰明":     4,
    "太棒了":     4,

    # ── Fear / danger ─────────────────────────────────────────────────────────
    "不要":  2,
    "怕怕":  3,
    "快跑":  3,
    "壞人":  2,
    "Bad":   2,

    # ── Child-specific cute words ─────────────────────────────────────────────
    "拜拜":  2,
    "叔叔":  1,
    "阿嬤":  2,
    "阿嬷":  2,
    "哥哥":  1,
    "姐姐":  1,
    "格格":  1,

    # ── Game events ───────────────────────────────────────────────────────────
    "打王":   4,
    "睡著":   2,
    "睡着":   2,
    "復活":   3,

    # ── Real-world intrusions (high interest!) ────────────────────────────────
    "地震":   6,
    "爆米花": 6,
    "卡片":   3,
    "擦桌子": 5,

    # ── Surprise / discovery ─────────────────────────────────────────────────
    "你看":   2,
    "快看":   3,
    "看看":   2,
    "有泳":   3,
    "游泳":   2,
    "下雪":   3,
    "打雷":   4,
}

# Keywords that should always trigger a highlight regardless of audio score
MUST_INCLUDE_KEYWORDS = {"地震", "爆米花", "擦桌子", "哇塞", "你好厲害", "太厲害"}


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization helper
# ═══════════════════════════════════════════════════════════════════════════════
def norm(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# Audio extraction
# ═══════════════════════════════════════════════════════════════════════════════
def extract_audio(video_path: str, audio_path: str):
    """Extract mono 16 kHz PCM audio (optimal format for Whisper + RMS analysis)."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", audio_path
    ], check=True, capture_output=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Whisper transcription
# ═══════════════════════════════════════════════════════════════════════════════
def transcribe(audio_path: str, model_name: str = "medium"):
    """Run Whisper and return the full result dict with segments."""
    print(f"  Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)
    print("  Transcribing audio (this may take several minutes)...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        language="zh",
        task="transcribe",
        initial_prompt="請使用繁體中文。以下是普通話對話。",
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SRT writing
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_WHISPER_TEMPERATURES = (0.0, 0.2, 0.4, 0.6)
DEFAULT_WHISPER_PROMPT = (
    "請使用繁體中文逐字轉寫。內容是爸爸 Ken 和女兒 Amelia 一起玩《隻狼》。"
    "常見詞：阿梅莉亞、Ken、Sekiro、隻狼、忍者、佛雕師、葦名、弦一郎。"
    "請保留語氣詞、笑聲、驚呼。"
)


def transcribe(
    audio_path: str,
    model_name: str = "medium",
    *,
    beam_size: int = 5,
    best_of: int = 5,
    temperatures=DEFAULT_WHISPER_TEMPERATURES,
    condition_on_previous_text: bool = False,
    initial_prompt: str = DEFAULT_WHISPER_PROMPT,
):
    """Run Whisper and return the full result dict with tuned decoding defaults."""
    print(f"  Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)
    print("  Transcribing audio (this may take several minutes)...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        language="zh",
        task="transcribe",
        beam_size=beam_size,
        best_of=best_of,
        temperature=tuple(temperatures),
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
    )
    return result


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h  = int(seconds) // 3600
    m  = (int(seconds) % 3600) // 60
    s  = int(seconds) % 60
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_full_srt(segments, srt_path: str):
    """Write the complete SRT file for the whole video."""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_fmt_time(seg['start'])} --> {_fmt_time(seg['end'])}\n"
                    f"{seg['text'].strip()}\n\n")


def write_clip_srt(segments, clip_start: float, clip_end: float, srt_path: str):
    """Write an SRT for a specific clip window, with timestamps re-zeroed to clip start."""
    entries = []
    for seg in segments:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue
        start_adj = max(0.0, seg["start"] - clip_start)
        end_adj   = max(0.0, seg["end"]   - clip_start)
        entries.append((start_adj, end_adj, seg["text"].strip()))

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(entries, 1):
            f.write(f"{i}\n{_fmt_time(s)} --> {_fmt_time(e)}\n{text}\n\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════
def compute_audio_scores(audio_path: str, duration: float) -> np.ndarray:
    """
    Per-second excitement ratio based on RMS amplitude.
    Returns an array of shape (n_seconds,) where high values = loud/exciting moments.
    """
    audio_raw = np.fromfile(audio_path, dtype=np.int16)
    audio = audio_raw.astype(np.float32) / 32768.0
    sr = 16000
    n  = int(len(audio) / sr)
    rms = np.array([
        np.sqrt(np.mean(audio[i * sr:(i + 1) * sr] ** 2))
        for i in range(n)
    ])
    kernel   = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
    smoothed = np.convolve(rms, kernel, mode="same")
    baseline = np.array([
        np.median(smoothed[max(0, i - 30):i + 30])
        for i in range(len(smoothed))
    ])
    ratio = smoothed / (baseline + 1e-6)
    return ratio


def compute_subtitle_scores(segments, duration: float):
    """
    Per-second score based on Traditional Chinese reaction keyword weights
    and dialogue density. Returns (scores_array, must_include_seconds_list).
    """
    n = int(duration) + 2
    keyword_scores  = np.zeros(n)
    density_scores  = np.zeros(n)
    must_include    = []

    for seg in segments:
        t    = int(seg["start"])
        text = seg["text"]

        if t >= n:
            continue

        for keyword, weight in REACTION_KEYWORDS.items():
            count = text.count(keyword)
            if count:
                keyword_scores[t] += weight * count

        for kw in MUST_INCLUDE_KEYWORDS:
            if kw in text:
                must_include.append(t)

        density_scores[t] += 1

    keyword_scores = np.convolve(keyword_scores, np.ones(5) / 5, mode="same")

    density_smooth = np.convolve(density_scores, np.ones(5) / 5, mode="same")
    if density_smooth.max() > 0:
        density_smooth = density_smooth / density_smooth.max() * 3

    scores = keyword_scores + density_smooth
    return scores, must_include


def combine_highlight_scores(
    audio_scores: np.ndarray,
    subtitle_scores: np.ndarray,
    amelia_scores: np.ndarray | None = None,
    *,
    weights: tuple[float, float, float] = (0.45, 0.55, 0.0),
) -> np.ndarray:
    """Combine normalized score streams into a single smoothed highlight score."""
    if amelia_scores is None:
        length = min(len(audio_scores), len(subtitle_scores))
    else:
        length = min(len(audio_scores), len(subtitle_scores), len(amelia_scores))

    if length == 0:
        return np.zeros(0, dtype=np.float32)

    audio_weight, subtitle_weight, amelia_weight = weights
    combined = audio_weight * norm(audio_scores[:length]) + subtitle_weight * norm(subtitle_scores[:length])
    if amelia_scores is not None and amelia_weight > 0:
        combined += amelia_weight * norm(amelia_scores[:length])
    kernel_size = min(5, length)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(combined, kernel, mode="same")


def pick_highlights(audio_scores: np.ndarray,
                    subtitle_scores: np.ndarray,
                    must_include: list,
                    n_clips=25,
                    min_gap: int = 15,
                    amelia_scores: np.ndarray | None = None,
                    weights: tuple[float, float, float] = (0.45, 0.55, 0.0)) -> list:
    """
    Combine audio + subtitle signals and pick top N non-overlapping highlights.
    Must-include keywords always get a clip regardless of score rank.
    Returns a list of (center_second, combined_score) sorted by timestamp.
    """
    import math
    combined = combine_highlight_scores(
        audio_scores,
        subtitle_scores,
        amelia_scores=amelia_scores,
        weights=weights,
    )
    length = len(combined)

    for t in must_include:
        if t < length:
            combined[t] = combined.max() * 1.5

    working = combined.copy()
    peaks   = []
    max_peaks = math.inf if n_clips == math.inf else int(n_clips)

    while len(peaks) < max_peaks:
        idx = int(np.argmax(working))
        if working[idx] < 0.01:
            break
        peaks.append((idx, float(combined[idx])))
        lo = max(0, idx - min_gap)
        hi = min(length, idx + min_gap)
        working[lo:hi] = 0

    return sorted(peaks, key=lambda x: x[0])


def compute_clip_bounds(center: int,
                        combined_scores: np.ndarray,
                        video_duration: float,
                        min_dur: int = 10,
                        max_dur: int = 60) -> tuple:
    """
    Compute smart start/end for a clip centered on `center` seconds.
    Returns (clip_start, clip_end) in seconds.
    """
    n = len(combined_scores)
    threshold = float(np.percentile(combined_scores, 60))

    back_limit = max(0, center - max_dur)
    start = center
    consecutive_low = 0
    for t in range(center, back_limit, -1):
        if t < n and combined_scores[t] >= threshold:
            start = t
            consecutive_low = 0
        else:
            consecutive_low += 1
            if consecutive_low > 1:
                break

    fwd_limit = min(n - 1, center + max_dur)
    end = center
    consecutive_low = 0
    for t in range(center, fwd_limit):
        if t < n and combined_scores[t] >= threshold:
            end = t
            consecutive_low = 0
        else:
            consecutive_low += 1
            if consecutive_low > 1:
                break

    actual = end - start
    if actual < min_dur:
        pad = (min_dur - actual) // 2
        start = max(0, start - pad)
        end   = min(int(video_duration), end + pad)
        if end - start < min_dur:
            end   = min(int(video_duration), start + min_dur)
            start = max(0, end - min_dur)

    if end - start > max_dur:
        half  = max_dur // 2
        start = max(0, center - half)
        end   = min(int(video_duration), start + max_dur)

    return float(start), float(end)


# ═══════════════════════════════════════════════════════════════════════════════
# Clip description
# ═══════════════════════════════════════════════════════════════════════════════
def describe_clip(segments, center: float, window: float = 8.0) -> str:
    """Build a short, filesystem-safe description from subtitle text near this timestamp."""
    nearby = [
        seg["text"].strip()
        for seg in segments
        if abs(seg["start"] - center) <= window
    ]
    if not nearby:
        return "clip"

    best = ""
    best_score = -1
    for text in nearby:
        score = sum(w for kw, w in REACTION_KEYWORDS.items() if kw in text)
        if score > best_score:
            best_score = score
            best = text

    if not best:
        best = nearby[0]

    clean = re.sub(r'\s+', '_', best.strip())
    clean = re.sub(r'[^\w\u4e00-\u9fff\u3000-\u303f]', '', clean)
    return clean[:28] if clean else "clip"


# ═══════════════════════════════════════════════════════════════════════════════
# Clip cutting
# ═══════════════════════════════════════════════════════════════════════════════
def cut_clip(video_path: str, start: float, duration: float, out_mp4: str):
    """Cut a raw clip (no subtitles yet)."""
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac",
        out_mp4
    ], check=True, capture_output=True)


def burn_subtitles(raw_mp4: str, srt_path: str, out_mp4: str):
    """Burn SRT subtitles into the video."""
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_mp4,
        "-vf", (
            f"subtitles='{srt_escaped}':force_style='"
            "FontName=Arial,"
            "FontSize=20,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BackColour=&H80000000,"
            "Outline=2,"
            "Shadow=1,"
            "Alignment=2"
            "'"
        ),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        out_mp4
    ], check=True, capture_output=True)
