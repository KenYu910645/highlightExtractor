#!/usr/bin/env python3
"""
preprocess.py - Step 1: Transcribe video and extract highlight candidates
=========================================================================
Given an MP4 video, this script:
  1. Transcribes audio with OpenAI Whisper to a full .srt subtitle file
  2. Scores every second using audio RMS + Traditional Chinese subtitle analysis
  3. Selects top N candidate highlight moments (generously, default 1000)
  4. Extracts a representative JPEG thumbnail for each candidate
  5. Writes candidates.md as a structured overview for AI review

Usage:
    python preprocess.py <video.mp4> [options]

Options:
    --candidates N     Number of candidate moments to extract (default: 1000)
    --min-gap N        Min seconds between candidates (default: 15)
    --model NAME       Whisper model name, e.g. small/medium/large (default: large)
    --beam-size N      Whisper beam size (default: 5)
    --best-of N        Whisper best_of setting (default: 5)
    --temperatures S   Whisper fallback temperatures (default: 0.0,0.2,0.4,0.6)
    --frame-size WxH   Thumbnail dimensions (default: 640x360)

Output (next to the video file):
    {stem}.srt
    {stem}_candidates/
        candidate_01_02m34s.jpg
        candidate_02_05m11s.jpg
        ...
        candidates.md
"""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import opencc

from utils import (
    DEFAULT_WHISPER_PROMPT,
    DEFAULT_WHISPER_TEMPERATURES,
    compute_audio_scores,
    compute_subtitle_scores,
    extract_audio,
    norm,
    pick_highlights,
    transcribe,
    write_full_srt,
)


def parse_temperatures(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of Whisper fallback temperatures."""
    try:
        temperatures = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--temperatures must be a comma-separated list like 0.0,0.2,0.4,0.6"
        ) from exc
    if not temperatures:
        raise argparse.ArgumentTypeError("--temperatures must include at least one float")
    return temperatures


def extract_frame(
    video_path: str,
    timestamp_sec: float,
    out_jpg: str,
    width: int = 640,
    height: int = 360,
):
    """Extract a single JPEG thumbnail from the video using fast keyframe seek."""
    scale_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp_sec),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            scale_filter,
            "-q:v",
            "3",
            out_jpg,
        ],
        check=True,
        capture_output=True,
    )


def compute_window_rms(audio_scores: np.ndarray, center: int, half: int = 10) -> float:
    """Mean normalized audio excitement around a candidate center."""
    lo = max(0, center - half)
    hi = min(len(audio_scores), center + half + 1)
    if lo >= hi:
        return 0.0
    arr_norm = norm(audio_scores)
    return float(np.mean(arr_norm[lo:hi]))


def gather_nearby_subtitles(segments: list, center: float, window: float = 12.0) -> str:
    """Collect nearby subtitle text for AI review."""
    nearby = []
    for seg in segments:
        if abs(seg["start"] - center) > window:
            continue
        text = seg["text"].strip()
        if text:
            nearby.append(text)
    if not nearby:
        return "(no dialogue)"
    joined = " / ".join(nearby)
    return joined[:300] if len(joined) > 300 else joined


def write_candidates_md(candidates: list, out_path: str, video_name: str, srt_path: str):
    """Write the markdown bundle that the AI review step consumes."""
    lines = [
        f"# Highlight Candidates - {video_name}",
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
        "* confidence: 0.00-1.00",
        "```",
        "",
        "You may adjust the start/end times freely. The candidate timestamp is",
        "just the peak moment; you decide how wide to make the clip.",
        "",
        "---",
        "",
    ]

    for candidate in candidates:
        ts = f"{candidate['mm']:02d}:{candidate['ss']:02d}"
        lines.extend(
            [
                f"## Candidate {candidate['index']:02d} - {ts}",
                f"- **Image:** {candidate['image_filename']}",
                f"- **Timestamp:** {ts} ({candidate['center_sec']:.0f} seconds)",
                f"- **Composite score:** {candidate['score']:.3f}",
                f"- **Audio excitement:** {candidate['audio_rms']:.3f}  (0=silent, 1=loudest)",
                f"- **Nearby dialogue:** {candidate['subtitle_text']}",
                "",
            ]
        )

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process video: transcribe and extract highlight candidate thumbnails"
    )
    parser.add_argument("video", help="Input MP4 file path")
    parser.add_argument(
        "--candidates",
        type=int,
        default=1000,
        help="Number of candidate moments (default: 1000)",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=15,
        help="Min seconds between candidates (default: 15)",
    )
    parser.add_argument(
        "--model",
        default="large",
        help="Whisper model name, e.g. small/medium/large (default: large)",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size (default: 5)")
    parser.add_argument("--best-of", type=int, default=5, help="Whisper best_of setting (default: 5)")
    parser.add_argument(
        "--temperatures",
        type=parse_temperatures,
        default=DEFAULT_WHISPER_TEMPERATURES,
        help="Whisper fallback temperatures, comma-separated (default: 0.0,0.2,0.4,0.6)",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_true",
        help="Let Whisper condition each segment on the previous decoded text",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Disable Whisper previous-text conditioning (default)",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_WHISPER_PROMPT,
        help="Prompt hint passed to Whisper before decoding",
    )
    parser.add_argument(
        "--frame-size",
        default="640x360",
        help="Thumbnail dimensions WxH (default: 640x360)",
    )
    parser.set_defaults(condition_on_previous_text=False)
    args = parser.parse_args()

    try:
        frame_w, frame_h = (int(x) for x in args.frame_size.lower().split("x"))
    except ValueError as exc:
        raise SystemExit(f"ERROR: --frame-size must be WxH, e.g. 640x360. Got: {args.frame_size!r}") from exc

    video_path = Path(args.video).resolve()
    stem = video_path.stem
    out_dir = video_path.parent
    full_srt = str(out_dir / f"{stem}.srt")
    candidates_dir = out_dir / f"{stem}_candidates"
    candidates_dir.mkdir(exist_ok=True)
    tmp_audio = str(out_dir / f"_tmp_{stem}_audio.wav")

    print(f"\n{'=' * 60}")
    print("  Pre-processor")
    print(f"  Video      : {video_path.name}")
    print(f"  Candidates : {args.candidates}  |  Gap: {args.min_gap}s")
    print(f"  Model      : Whisper {args.model}")
    print(
        f"  Decode     : beam={args.beam_size}, best_of={args.best_of}, "
        f"temps={','.join(f'{temp:g}' for temp in args.temperatures)}, "
        f"prev_text={'On' if args.condition_on_previous_text else 'Off'}"
    )
    print(f"  Thumbnails : {frame_w}x{frame_h}")
    print(f"{'=' * 60}\n")

    print("[1/5] Extracting audio...")
    extract_audio(str(video_path), tmp_audio)
    print("  Done\n")

    print("[2/5] Transcribing with Whisper...")
    result = transcribe(
        tmp_audio,
        args.model,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperatures=args.temperatures,
        condition_on_previous_text=args.condition_on_previous_text,
        initial_prompt=args.initial_prompt,
    )
    segments = result["segments"]
    duration = segments[-1]["end"] if segments else 0

    converter = opencc.OpenCC("s2twp")
    for seg in segments:
        seg["text"] = converter.convert(seg["text"])

    print(f"  {len(segments)} segments | {duration / 60:.1f} min")
    print("  Converted to Traditional Chinese\n")

    print("[3/5] Writing full subtitle file...")
    write_full_srt(segments, full_srt)
    print(f"  {Path(full_srt).name}\n")

    print("[4/5] Scoring and selecting candidates...")
    audio_scores = compute_audio_scores(tmp_audio, duration)
    subtitle_scores, must_include = compute_subtitle_scores(segments, duration)

    length = min(len(audio_scores), len(subtitle_scores))
    _combined_scores = np.convolve(
        0.45 * norm(audio_scores[:length]) + 0.55 * norm(subtitle_scores[:length]),
        np.ones(5) / 5,
        mode="same",
    )
    candidates_raw = pick_highlights(
        audio_scores,
        subtitle_scores,
        must_include,
        n_clips=args.candidates,
        min_gap=args.min_gap,
    )
    print(f"  {len(candidates_raw)} candidates selected ({len(must_include)} must-include forced)\n")

    os.remove(tmp_audio)

    print("[5/5] Extracting thumbnails and writing candidates.md...")
    candidates = []
    for i, (center, score) in enumerate(candidates_raw, 1):
        mm = int(center) // 60
        ss = int(center) % 60
        img_name = f"candidate_{i:02d}_{mm:02d}m{ss:02d}s.jpg"
        img_path = str(candidates_dir / img_name)

        try:
            extract_frame(str(video_path), float(center), img_path, frame_w, frame_h)
        except subprocess.CalledProcessError:
            print(
                f"  WARNING: frame extraction failed for candidate {i:02d} at {mm:02d}:{ss:02d}"
            )
            img_name = "(extraction failed)"

        candidate = {
            "index": i,
            "center_sec": float(center),
            "mm": mm,
            "ss": ss,
            "score": score,
            "audio_rms": compute_window_rms(audio_scores, center),
            "subtitle_text": gather_nearby_subtitles(segments, float(center)),
            "image_filename": img_name,
        }
        candidates.append(candidate)
        print(f"  [{i:02d}] {mm:02d}:{ss:02d}  score={score:.3f}  -> {img_name}")

    candidates_md = str(candidates_dir / "candidates.md")
    write_candidates_md(candidates, candidates_md, video_path.name, full_srt)

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"  SRT        : {Path(full_srt).name}")
    print(f"  Candidates : {candidates_dir.name}/  ({len(candidates)} thumbnails)")
    print(f"  AI context : {candidates_dir.name}/candidates.md")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
