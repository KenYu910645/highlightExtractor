#!/usr/bin/env python3
"""
highlight_extractor.py — Automated video highlight pipeline
============================================================
Given an MP4 video, this script will:
  1. Transcribe audio with OpenAI Whisper  → full .srt subtitle file
  2. Score every second using audio RMS + Traditional Chinese subtitle analysis
  3. Pick the top highlight moments (erring on the side of too many)
  4. Cut each highlight clip with VARIABLE duration (10–60 s, extended to capture buildup)
  5. Burn subtitles directly into each clip + save a companion .srt

Per AGENTS.md policy: prefer too many clips over missing a funny moment.
Clip length auto-extends to cover natural conversation buildup; never cuts short.

Usage:
    python3 highlight_extractor.py <video.mp4> [options]

Options:
    --clips N          Number of highlight clips (default: unlimited; pass a number to limit)
    --min-gap N        Minimum seconds between clip centers (default: 15)
    --min-dur N        Minimum clip duration in seconds     (default: 10)
    --max-dur N        Maximum clip duration in seconds     (default: 60)
    --model NAME       Whisper model: tiny / small / medium (default: medium)
    --no-burn          Skip subtitle burning (clips will still have .srt files)

Examples:
    python3 highlight_extractor.py Day4.mp4
    python3 highlight_extractor.py Day5.mp4 --clips 20 --model small
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

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
# Adjust weights here to tune what counts as "interesting"
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
# STEP 1 — Audio extraction
# ═══════════════════════════════════════════════════════════════════════════════
def extract_audio(video_path: str, audio_path: str):
    """Extract mono 16 kHz PCM audio (optimal format for Whisper + RMS analysis)."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", audio_path
    ], check=True, capture_output=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Whisper transcription
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
        initial_prompt="請使用繁體中文。以下是普通話對話。",  # nudge Whisper toward Traditional Chinese
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SRT writing (full video + per-clip with shifted timestamps)
# ═══════════════════════════════════════════════════════════════════════════════
def transcribe(audio_path: str, model_name: str = "medium"):
    """Run Whisper and return the full result dict with tuned decoding defaults."""
    print(f"  Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)
    print("  Transcribing audio (this may take several minutes)...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        language="zh",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6),
        condition_on_previous_text=False,
        initial_prompt=(
            "請使用繁體中文逐字轉寫。內容是爸爸 Ken 和女兒 Amelia 一起玩《隻狼》。"
            "常見詞：阿梅莉亞、Ken、Sekiro、隻狼、忍者、佛雕師、葦名、弦一郎。"
            "請保留語氣詞、笑聲、驚呼。"
        ),
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
# STEP 4 — Scoring
# ═══════════════════════════════════════════════════════════════════════════════
def compute_audio_scores(audio_path: str, duration: float) -> np.ndarray:
    """
    Per-second excitement ratio based on RMS amplitude.
    Returns an array of shape (n_seconds,) where high values = loud/exciting moments.
    """
    audio_raw = np.fromfile(audio_path, dtype=np.int16)
    audio = audio_raw.astype(np.float32) / 32768.0  # normalize int16 PCM to [-1.0, 1.0]
    sr = 16000
    n  = int(len(audio) / sr)
    rms = np.array([
        np.sqrt(np.mean(audio[i * sr:(i + 1) * sr] ** 2))
        for i in range(n)
    ])
    # Smooth with 5-sample Gaussian kernel (reduces ringing at sharp transients)
    kernel   = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
    smoothed = np.convolve(rms, kernel, mode="same")
    # Rolling 60-second median baseline
    baseline = np.array([
        np.median(smoothed[max(0, i - 30):i + 30])
        for i in range(len(smoothed))
    ])
    ratio = smoothed / (baseline + 1e-6)
    return ratio


def compute_subtitle_scores(segments, duration: float) -> np.ndarray:
    """
    Per-second score based on:
      a) Traditional Chinese reaction keyword weights
      b) Dialogue density (fast back-and-forth = high energy)
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

        # Keyword scoring
        for keyword, weight in REACTION_KEYWORDS.items():
            count = text.count(keyword)
            if count:
                keyword_scores[t] += weight * count

        # Check for must-include keywords
        for kw in MUST_INCLUDE_KEYWORDS:
            if kw in text:
                must_include.append(t)

        # Dialogue density: +1 per segment
        density_scores[t] += 1

    # Smooth keyword scores over ±2 seconds
    keyword_scores = np.convolve(keyword_scores, np.ones(5) / 5, mode="same")

    # Normalize density to 0–3 range
    density_smooth = np.convolve(density_scores, np.ones(5) / 5, mode="same")
    if density_smooth.max() > 0:
        density_smooth = density_smooth / density_smooth.max() * 3

    scores = keyword_scores + density_smooth
    return scores, must_include


def pick_highlights(audio_scores: np.ndarray,
                    subtitle_scores: np.ndarray,
                    must_include: list,
                    n_clips: int = 25,
                    min_gap: int = 15) -> list:
    """
    Combine audio + subtitle signals and pick top N non-overlapping highlights.
    Must-include keywords always get a clip regardless of score rank.
    Returns a list of (center_second, combined_score) sorted by timestamp.

    Per AGENTS.md: n_clips defaults to 25 and min_gap is 15 s — erring on the
    side of too many clips so Ken can manually cherry-pick the best ones.
    """
    length = min(len(audio_scores), len(subtitle_scores))

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    combined = (0.45 * norm(audio_scores[:length]) +
                0.55 * norm(subtitle_scores[:length]))  # subtitle weighted slightly higher
    combined = np.convolve(combined, np.ones(5) / 5, mode="same")

    # Force must-include moments to maximum score
    for t in must_include:
        if t < length:
            combined[t] = combined.max() * 1.5

    # Greedy peak selection with minimum gap
    working = combined.copy()
    peaks   = []

    while len(peaks) < n_clips:
        idx = int(np.argmax(working))
        if working[idx] < 0.01:
            break
        peaks.append((idx, float(combined[idx])))
        # Suppress neighbourhood
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

    Strategy (per AGENTS.md — never cut short, extend to capture buildup):
      - Walk backward from center until score drops below threshold OR max_dur reached
      - Walk forward  from center until score drops below threshold OR max_dur reached
      - Enforce minimum duration by padding symmetrically if needed
      - Hard cap at max_dur seconds total

    Returns (clip_start, clip_end) in seconds.
    """
    n = len(combined_scores)
    threshold = float(np.percentile(combined_scores, 60))  # "above average" activity

    # Walk backward to find natural start of moment
    # Tolerates up to 1 consecutive below-threshold second to bridge brief dips
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

    # Walk forward to find natural end of moment (same dip tolerance)
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

    # Enforce minimum duration — pad symmetrically
    actual = end - start
    if actual < min_dur:
        pad = (min_dur - actual) // 2
        start = max(0, start - pad)
        end   = min(int(video_duration), end + pad)
        # If still short (near video edge), extend the other direction
        if end - start < min_dur:
            end   = min(int(video_duration), start + min_dur)
            start = max(0, end - min_dur)

    # Hard cap at max_dur
    if end - start > max_dur:
        half  = max_dur // 2
        start = max(0, center - half)
        end   = min(int(video_duration), start + max_dur)

    return float(start), float(end)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Clip description from subtitle text
# ═══════════════════════════════════════════════════════════════════════════════
def describe_clip(segments, center: float, window: float = 8.0) -> str:
    """
    Build a short, filesystem-safe description from subtitle text near this timestamp.
    Prioritises segments containing high-weight reaction keywords.
    """
    nearby = [
        seg["text"].strip()
        for seg in segments
        if abs(seg["start"] - center) <= window
    ]
    if not nearby:
        return "clip"

    # Find segment with highest keyword match
    best = ""
    best_score = -1
    for text in nearby:
        score = sum(w for kw, w in REACTION_KEYWORDS.items() if kw in text)
        if score > best_score:
            best_score = score
            best = text

    # If nothing scored, just use first segment
    if not best:
        best = nearby[0]

    # Sanitize for filename
    clean = re.sub(r'\s+', '_', best.strip())
    clean = re.sub(r'[^\w\u4e00-\u9fff\u3000-\u303f]', '', clean)
    return clean[:28] if clean else "clip"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Clip cutting with burned-in subtitles
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
    """
    Burn SRT subtitles into the video.
    Font style: white text, black outline, bottom-center, readable size.
    """
    # ffmpeg subtitles filter needs the srt path escaped for special characters
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_mp4,
        "-vf", (
            f"subtitles='{srt_escaped}':force_style='"
            "FontName=Arial,"
            "FontSize=20,"
            "PrimaryColour=&H00FFFFFF,"   # white text
            "OutlineColour=&H00000000,"   # black outline
            "BackColour=&H80000000,"      # semi-transparent background
            "Outline=2,"
            "Shadow=1,"
            "Alignment=2"                 # bottom-center
            "'"
        ),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        out_mp4
    ], check=True, capture_output=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Automated video highlight extractor with Traditional Chinese subtitle analysis"
    )
    def _clips_arg(v):
        return math.inf if v.lower() in ("inf", "unlimited", "0") else int(v)

    parser.add_argument("video",           help="Input MP4 file path")
    parser.add_argument("--clips",         type=_clips_arg, default=math.inf, help="Number of highlight clips (default: unlimited)")
    parser.add_argument("--min-gap",       type=int,   default=15,      help="Min seconds between clips (default: 15)")
    parser.add_argument("--min-dur",       type=int,   default=10,      help="Min clip duration in seconds (default: 10)")
    parser.add_argument("--max-dur",       type=int,   default=60,      help="Max clip duration in seconds (default: 60)")
    parser.add_argument("--model",         default="medium",            help="Whisper model name, e.g. small/medium/large (default: medium)")
    parser.add_argument("--no-burn",       action="store_true",         help="Skip subtitle burning into clips")
    args = parser.parse_args()

    video_path    = Path(args.video).resolve()
    out_dir       = video_path.parent
    stem          = video_path.stem
    highlight_dir = out_dir / "highlight"
    highlight_dir.mkdir(exist_ok=True)

    tmp_audio = str(out_dir / f"_tmp_{stem}_audio.wav")
    full_srt  = str(out_dir / f"{stem}.srt")

    print(f"\n{'='*60}")
    print(f"  🎬  Highlight Extractor")
    print(f"  Video   : {video_path.name}")
    clips_display = "unlimited" if args.clips == math.inf else str(int(args.clips))
    print(f"  Clips   : {clips_display}  |  Gap: {args.min_gap}s  |  Dur: {args.min_dur}–{args.max_dur}s")
    print(f"  Model   : Whisper {args.model}")
    print(f"  Burn-in : {'No' if args.no_burn else 'Yes'}")
    print(f"{'='*60}\n")

    # ── 1. Extract audio ──────────────────────────────────────────────────────
    print("[1/5] Extracting audio...")
    extract_audio(str(video_path), tmp_audio)
    print("  ✓ Done\n")

    # ── 2. Transcribe ─────────────────────────────────────────────────────────
    print("[2/5] Transcribing with Whisper...")
    result   = transcribe(tmp_audio, args.model)
    segments = result["segments"]
    duration = segments[-1]["end"] if segments else 0
    print(f"  ✓ {len(segments)} segments | {duration/60:.1f} min | Language: {result['language']}")

    # Convert any Simplified Chinese characters to Traditional Chinese
    converter = opencc.OpenCC("s2twp")
    for seg in segments:
        seg["text"] = converter.convert(seg["text"])
    print("  ✓ Converted to Traditional Chinese (opencc s2twp)\n")

    # ── 3. Write full SRT ─────────────────────────────────────────────────────
    print("[3/5] Writing full subtitle file...")
    write_full_srt(segments, full_srt)
    print(f"  ✓ {Path(full_srt).name}\n")

    # ── 4. Score & pick highlights ────────────────────────────────────────────
    print("[4/5] Scoring moments...")
    audio_scores              = compute_audio_scores(tmp_audio, duration)
    subtitle_scores, must_inc = compute_subtitle_scores(segments, duration)

    # Build combined score array (needed for smart clip bounds)
    length = min(len(audio_scores), len(subtitle_scores))
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)
    combined_scores = (0.45 * norm(audio_scores[:length]) +
                       0.55 * norm(subtitle_scores[:length]))
    combined_scores = np.convolve(combined_scores, np.ones(5) / 5, mode="same")

    highlights = pick_highlights(
        audio_scores, subtitle_scores, must_inc,
        n_clips=args.clips, min_gap=args.min_gap
    )
    print(f"  ✓ {len(highlights)} highlights selected")
    print(f"  ✓ {len(must_inc)} must-include moments forced in\n")

    # Cleanup temp audio
    os.remove(tmp_audio)

    # ── 5. Cut clips ──────────────────────────────────────────────────────────
    print(f"[5/5] Cutting clips {'with burned subtitles' if not args.no_burn else '(no burn)'}...")

    for i, (center, score) in enumerate(highlights, 1):
        # Smart variable-length bounds based on score activity
        clip_start, clip_end = compute_clip_bounds(
            center, combined_scores, duration,
            min_dur=args.min_dur, max_dur=args.max_dur
        )
        clip_duration = clip_end - clip_start

        mm   = int(center) // 60
        ss   = int(center) % 60
        desc = describe_clip(segments, float(center))
        name = f"{i:02d}_{mm:02d}m{ss:02d}s_{desc}"

        clip_srt = str(highlight_dir / f"{name}.srt")
        clip_mp4 = str(highlight_dir / f"{name}.mp4")
        raw_mp4  = str(highlight_dir / f"_raw_{name}.mp4")

        # Write clip-specific SRT
        write_clip_srt(segments, clip_start, clip_end, clip_srt)

        if args.no_burn:
            cut_clip(str(video_path), clip_start, clip_duration, clip_mp4)
        else:
            cut_clip(str(video_path), clip_start, clip_duration, raw_mp4)
            try:
                burn_subtitles(raw_mp4, clip_srt, clip_mp4)
                os.remove(raw_mp4)
            except subprocess.CalledProcessError:
                os.rename(raw_mp4, clip_mp4)
                print(f"  ⚠  Subtitle burn failed for clip {i:02d}, saved without burn")

        print(f"  ✓ [{i:02d}] {mm:02d}:{ss:02d}  {clip_duration:.0f}s  score={score:.3f}  → {name}.mp4")

    print(f"\n{'='*60}")
    print(f"  ✅  All done!")
    print(f"  📄  Subtitle : {Path(full_srt).name}")
    print(f"  📁  Clips    : highlight/  ({len(highlights)} clips)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
