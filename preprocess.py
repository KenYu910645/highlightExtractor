#!/usr/bin/env python3
"""
preprocess.py — Step 1: Transcribe video and extract highlight candidates
=========================================================================
Given an MP4 video, this script:
  1. Transcribes audio with OpenAI Whisper  → full .srt subtitle file
  2. Scores every second using audio RMS + Traditional Chinese subtitle analysis
  3. Selects top N candidate highlight moments (generously, default 1000)
  4. Extracts a representative JPEG thumbnail for each candidate
  5. Writes candidates.md — a structured overview for AI review (Step 2)

Usage:
    python preprocess.py <video.mp4> [options]

Options:
    --candidates N     Number of candidate moments to extract (default: 1000)
    --min-gap N        Min seconds between candidates (default: 15)
    --model NAME       Whisper model: tiny / small / medium (default: medium)
    --frame-size WxH   Thumbnail dimensions (default: 640x360)

Output (next to the video file):
    {stem}.srt                    — full transcription
    {stem}_candidates/
        candidate_01_02m34s.jpg   — thumbnail at each candidate moment
        candidate_02_05m11s.jpg
        ...
        candidates.md             — metadata for AI review
"""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import opencc

from speaker_labeler import SpeakerLabelingConfig, SpeakerLabelingError, SpeakerLabelingPipeline
from utils import (
    extract_audio,
    transcribe,
    write_full_srt,
    compute_audio_scores,
    compute_subtitle_scores,
    pick_highlights,
    norm,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Frame extraction
# ═══════════════════════════════════════════════════════════════════════════════
def extract_frame(video_path: str, timestamp_sec: float, out_jpg: str,
                  width: int = 640, height: int = 360):
    """
    Extract a single JPEG thumbnail from the video at the given timestamp.
    Scales to width x height with letterbox/pillarbox padding to preserve aspect ratio.
    Uses fast seek (-ss before -i) — keyframe accuracy is fine for thumbnails.
    """
    scale_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", scale_filter,
        "-q:v", "3",
        out_jpg,
    ], check=True, capture_output=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate metadata helpers
# ═══════════════════════════════════════════════════════════════════════════════
def compute_window_rms(audio_scores: np.ndarray, center: int, half: int = 10) -> float:
    """Mean of normalized audio_scores within [center-half, center+half]. Returns float in [0,1]."""
    lo  = max(0, center - half)
    hi  = min(len(audio_scores), center + half + 1)
    window = audio_scores[lo:hi]
    if len(window) == 0:
        return 0.0
    # audio_scores is already a ratio array; normalize within [0,1] for display
    arr_norm = norm(audio_scores)
    return float(np.mean(arr_norm[lo:hi]))


def gather_nearby_subtitles(segments: list, center: float, window: float = 12.0) -> str:
    """
    Collect subtitle text from segments within ±window seconds of center.
    Returns segments joined by ' / ', capped at 300 characters.
    """
    nearby = []
    for seg in segments:
        if abs(seg["start"] - center) > window:
            continue
        text = seg["text"].strip()
        if not text:
            continue
        nearby.append(format_speaker_line(seg.get("speaker"), text))
    if not nearby:
        return "(no dialogue)"
    joined = " / ".join(nearby)
    return joined[:300] if len(joined) > 300 else joined


def format_speaker_line(speaker: str | None, text: str) -> str:
    """Format subtitle text for AI review, including speaker labels when available."""
    clean_text = text.strip()
    if not clean_text:
        return ""
    if not speaker:
        return clean_text
    return f"{speaker}: {clean_text}"


def write_speaker_segments_json(segments: list, out_path: str):
    """Write a machine-readable sidecar with per-segment speaker metadata."""
    payload = []
    for seg in segments:
        payload.append({
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": seg["text"].strip(),
            "speaker": seg.get("speaker", "unknown"),
            "speaker_confidence": float(seg.get("speaker_confidence", 0.0) or 0.0),
            "speaker_voice_sec": float(seg.get("speaker_voice_sec", 0.0) or 0.0),
            "speaker_scores": seg.get("speaker_scores", {}),
            "speaker_smoothed": bool(seg.get("speaker_smoothed", False)),
        })
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# candidates.md writer
# ═══════════════════════════════════════════════════════════════════════════════
def write_candidates_md(candidates: list, out_path: str,
                        video_name: str, srt_path: str):
    """
    Write a structured Markdown file listing all candidates for AI review.

    Each candidate section:
        ## Candidate 01 — 02:34
        - **Image:** candidate_01_02m34s.jpg
        - **Timestamp:** 02:34 (154 seconds)
        - **Composite score:** 0.847
        - **Audio excitement:** 0.723  (0=silent, 1=loudest)
        - **Nearby dialogue:** 哇塞！你好厲害 / 爸爸你看 / 快跑快跑
    """
    lines = [
        f"# Highlight Candidates — {video_name}",
        "",
        f"- **Source video:** `{video_name}`",
        f"- **Full subtitles:** `{Path(srt_path).name}`",
        f"- **Total candidates:** {len(candidates)}",
        "",
        "Review the images and subtitles below. For each candidate you want to",
        "include as a highlight clip, output a section in this exact format:",
        "",
        "```",
        "## highlight_01",
        "* start: MM:SS",
        "* end: MM:SS",
        "* reason: brief description",
        "* confidence: 0.00–1.00",
        "```",
        "",
        "You may adjust the start/end times freely — the candidate timestamp is",
        "just the peak moment; you decide how wide to make the clip.",
        "",
        "---",
        "",
    ]

    for c in candidates:
        mm  = c["mm"]
        ss  = c["ss"]
        ts  = f"{mm:02d}:{ss:02d}"
        idx = c["index"]

        lines += [
            f"## Candidate {idx:02d} — {ts}",
            f"- **Image:** {c['image_filename']}",
            f"- **Timestamp:** {ts} ({c['center_sec']:.0f} seconds)",
            f"- **Composite score:** {c['score']:.3f}",
            f"- **Audio excitement:** {c['audio_rms']:.3f}  (0=silent, 1=loudest)",
            f"- **Nearby dialogue:** {c['subtitle_text']}",
            "",
        ]

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pre-process video: transcribe and extract highlight candidate thumbnails"
    )
    parser.add_argument("video",          help="Input MP4 file path")
    parser.add_argument("--candidates",   type=int, default=1000,    help="Number of candidate moments (default: 1000)")
    parser.add_argument("--min-gap",      type=int, default=15,      help="Min seconds between candidates (default: 15)")
    parser.add_argument("--model",        default="medium",          help="Whisper model: tiny/small/medium (default: medium)")
    parser.add_argument("--frame-size",   default="640x360",         help="Thumbnail dimensions WxH (default: 640x360)")
    parser.add_argument("--speaker-labels",    dest="speaker_labels", action="store_true",
                        help="Enable offline speaker labeling (default: enabled)")
    parser.add_argument("--no-speaker-labels", dest="speaker_labels", action="store_false",
                        help="Disable offline speaker labeling")
    parser.add_argument("--enroll-dir",        default=str(project_root / "data" / "enroll"),
                        help="Enrollment directory with ken/ and amelia/ subfolders")
    parser.add_argument("--speaker-model",     default="speechbrain-ecapa",
                        help="Speaker embedding model name (default: speechbrain-ecapa)")
    parser.add_argument("--speaker-threshold", type=float, default=0.58,
                        help="Minimum similarity required before using a known speaker label")
    parser.add_argument("--min-voice-sec",     type=float, default=0.35,
                        help="Minimum voiced speech duration required to classify a subtitle segment")
    parser.set_defaults(speaker_labels=True)
    args = parser.parse_args()

    # Parse frame size
    try:
        frame_w, frame_h = (int(x) for x in args.frame_size.lower().split("x"))
    except ValueError:
        print(f"ERROR: --frame-size must be WxH, e.g. 640x360. Got: {args.frame_size!r}")
        raise SystemExit(1)

    video_path     = Path(args.video).resolve()
    stem           = video_path.stem
    out_dir        = video_path.parent
    full_srt       = str(out_dir / f"{stem}.srt")
    speaker_json   = str(out_dir / f"{stem}.speakers.json")
    candidates_dir = out_dir / f"{stem}_candidates"
    candidates_dir.mkdir(exist_ok=True)
    speaker_json_written = False

    tmp_audio = str(out_dir / f"_tmp_{stem}_audio.wav")

    print(f"\n{'='*60}")
    print(f"  Pre-processor")
    print(f"  Video      : {video_path.name}")
    print(f"  Candidates : {args.candidates}  |  Gap: {args.min_gap}s")
    print(f"  Model      : Whisper {args.model}")
    print(f"  Thumbnails : {frame_w}x{frame_h}")
    print(f"  Speakers   : {'On' if args.speaker_labels else 'Off'}")
    print(f"{'='*60}\n")

    # ── 1. Extract audio ──────────────────────────────────────────────────────
    print("[1/6] Extracting audio...")
    extract_audio(str(video_path), tmp_audio)
    print("  Done\n")

    # ── 2. Transcribe ─────────────────────────────────────────────────────────
    print("[2/6] Transcribing with Whisper...")
    result   = transcribe(tmp_audio, args.model)
    segments = result["segments"]
    duration = segments[-1]["end"] if segments else 0

    # Convert Simplified → Traditional Chinese
    converter = opencc.OpenCC("s2twp")
    for seg in segments:
        seg["text"] = converter.convert(seg["text"])

    print(f"  {len(segments)} segments | {duration/60:.1f} min")
    print("  Converted to Traditional Chinese\n")

    # ── 3. Write full SRT ─────────────────────────────────────────────────────
    if args.speaker_labels:
        print("[3/6] Classifying subtitle speakers...")
        try:
            speaker_config = SpeakerLabelingConfig(
                enroll_dir=Path(args.enroll_dir).resolve(),
                speaker_model=args.speaker_model,
                speaker_threshold=args.speaker_threshold,
                min_voice_sec=args.min_voice_sec,
            )
            speaker_pipeline = SpeakerLabelingPipeline(speaker_config)
            segments = speaker_pipeline.classify_segments(tmp_audio, segments)
            labeled_count = sum(1 for seg in segments if seg.get("speaker") in {"Ken", "Amelia"})
            unknown_count = sum(1 for seg in segments if seg.get("speaker") == "unknown")
            write_speaker_segments_json(segments, speaker_json)
            speaker_json_written = True
            print(f"  {Path(speaker_json).name}")
            print(f"  Known speakers: {labeled_count} segments | unknown: {unknown_count}\n")
        except (SpeakerLabelingError, RuntimeError) as exc:
            print(f"  WARNING: speaker labeling skipped: {exc}\n")

    print("[4/6] Writing full subtitle file...")
    write_full_srt(segments, full_srt)
    print(f"  {Path(full_srt).name}\n")

    # ── 4. Score and pick candidates ──────────────────────────────────────────
    print("[5/6] Scoring and selecting candidates...")
    audio_scores              = compute_audio_scores(tmp_audio, duration)
    subtitle_scores, must_inc = compute_subtitle_scores(segments, duration)

    length = min(len(audio_scores), len(subtitle_scores))
    combined_scores = (0.45 * norm(audio_scores[:length]) +
                       0.55 * norm(subtitle_scores[:length]))
    combined_scores = np.convolve(combined_scores, np.ones(5) / 5, mode="same")

    candidates_raw = pick_highlights(
        audio_scores, subtitle_scores, must_inc,
        n_clips=args.candidates, min_gap=args.min_gap
    )
    print(f"  {len(candidates_raw)} candidates selected ({len(must_inc)} must-include forced)\n")

    # Cleanup temp audio
    os.remove(tmp_audio)

    # ── 5. Extract frames and build metadata ─────────────────────────────────
    print("[6/6] Extracting thumbnails and writing candidates.md...")
    candidates = []

    for i, (center, score) in enumerate(candidates_raw, 1):
        mm  = int(center) // 60
        ss  = int(center) % 60
        img_name = f"candidate_{i:02d}_{mm:02d}m{ss:02d}s.jpg"
        img_path = str(candidates_dir / img_name)

        try:
            extract_frame(str(video_path), float(center), img_path, frame_w, frame_h)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: frame extraction failed for candidate {i:02d} at {mm:02d}:{ss:02d} - skipping thumbnail")
            img_name = "(extraction failed)"

        audio_rms    = compute_window_rms(audio_scores, center)
        subtitle_txt = gather_nearby_subtitles(segments, float(center))

        candidates.append({
            "index":          i,
            "center_sec":     float(center),
            "mm":             mm,
            "ss":             ss,
            "score":          score,
            "audio_rms":      audio_rms,
            "subtitle_text":  subtitle_txt,
            "image_filename": img_name,
        })

        print(f"  [{i:02d}] {mm:02d}:{ss:02d}  score={score:.3f}  -> {img_name}")

    candidates_md = str(candidates_dir / "candidates.md")
    write_candidates_md(candidates, candidates_md, video_path.name, full_srt)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  SRT        : {Path(full_srt).name}")
    if speaker_json_written:
        print(f"  Speakers   : {Path(speaker_json).name}")
    print(f"  Candidates : {candidates_dir.name}/  ({len(candidates)} thumbnails)")
    print(f"  AI context : {candidates_dir.name}/candidates.md")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
