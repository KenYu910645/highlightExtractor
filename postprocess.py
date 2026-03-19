#!/usr/bin/env python3
"""
postprocess.py - Step 3: Parse AI highlight markdown and cut clips.

Reads the AI-generated markdown (output of Step 2) and cuts each
approved highlight from the source video.

Usage:
    python postprocess.py <video.mp4> <highlights.md> [options]

Options:
    --srt PATH           Full video SRT for subtitle burning (default: <video_stem>.srt)
    --min-confidence F   Skip highlights below this confidence (default: 0.0)
    --no-burn            Cut clips without burning subtitles
    --out-dir PATH       Output directory (default: highlight/ next to video)
    --min-dur N          Minimum clip duration in seconds (default: 10)
    --max-dur N          Maximum clip duration in seconds (default: 60)

Expected markdown format (from AI Step 2):
    ## highlight_01
    * start: 02:34
    * end: 02:58
    * reason: everyone look good
    * confidence: 0.95
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

from utils import burn_subtitles, cut_clip, write_clip_srt


def parse_timestamp(ts: str) -> float:
    """Convert MM:SS or HH:MM:SS string to total seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unrecognized timestamp: {ts!r}")


def parse_highlight_md(md_path: str) -> list:
    """
    Parse AI-generated markdown into a list of highlight dicts.

    Each dict: index, start_sec, end_sec, start_str, end_str, reason, confidence.
    Blocks missing start or end are skipped with a printed warning.
    Returns list sorted by start_sec ascending.
    """
    text = Path(md_path).read_text(encoding="utf-8")
    blocks = re.split(r"^##\s+highlight_(\d+)", text, flags=re.IGNORECASE | re.MULTILINE)

    highlights = []
    pairs = list(zip(blocks[1::2], blocks[2::2]))

    for raw_index, body in pairs:
        index = int(raw_index)

        start_m = re.search(r"^[*\-]\s*start:\s*(.+)$", body, re.IGNORECASE | re.MULTILINE)
        end_m = re.search(r"^[*\-]\s*end:\s*(.+)$", body, re.IGNORECASE | re.MULTILINE)

        if not start_m or not end_m:
            print(f"  WARNING: highlight_{index:02d} missing start/end - skipping")
            continue

        try:
            start_str = start_m.group(1).strip()
            end_str = end_m.group(1).strip()
            start_sec = parse_timestamp(start_str)
            end_sec = parse_timestamp(end_str)
        except ValueError as exc:
            print(f"  WARNING: highlight_{index:02d} bad timestamp ({exc}) - skipping")
            continue

        reason_m = re.search(r"^[*\-]\s*reason:\s*(.+)$", body, re.IGNORECASE | re.MULTILINE)
        confidence_m = re.search(r"^[*\-]\s*confidence:\s*(.+)$", body, re.IGNORECASE | re.MULTILINE)

        reason = reason_m.group(1).strip() if reason_m else ""
        try:
            confidence = float(confidence_m.group(1).strip()) if confidence_m else 1.0
        except ValueError:
            confidence = 1.0

        highlights.append(
            {
                "index": index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_str": start_str,
                "end_str": end_str,
                "reason": reason,
                "confidence": confidence,
            }
        )

    return sorted(highlights, key=lambda h: h["start_sec"])


def make_clip_name(index: int, start_sec: float, reason: str) -> str:
    """Build a filesystem-safe clip filename stem from index, timestamp, and reason."""
    mm = int(start_sec) // 60
    ss = int(start_sec) % 60
    slug = re.sub(r"\s+", "_", reason.strip())
    slug = re.sub(r"[^\w\u4e00-\u9fff\u3000-\u303f]", "", slug)
    slug = slug[:40] if slug else "clip"
    return f"{index:02d}_{mm:02d}m{ss:02d}s_{slug}"


def clamp_duration(
    start_sec: float,
    end_sec: float,
    min_dur: int,
    max_dur: int,
    video_duration: float,
) -> tuple:
    """Enforce min/max duration. Pad symmetrically if short; truncate from end if long."""
    dur = end_sec - start_sec

    if dur < min_dur:
        pad = (min_dur - dur) / 2
        start_sec = max(0.0, start_sec - pad)
        end_sec = min(video_duration, end_sec + pad)
        if end_sec - start_sec < min_dur:
            end_sec = min(video_duration, start_sec + min_dur)
            start_sec = max(0.0, end_sec - min_dur)

    if end_sec - start_sec > max_dur:
        end_sec = start_sec + max_dur

    return start_sec, end_sec


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Post-process AI highlight markdown into video clips"
    )
    parser.add_argument("video", help="Source MP4 file path")
    parser.add_argument("highlights_md", help="AI-generated highlights markdown file")
    parser.add_argument(
        "--srt",
        default=None,
        help="Full video SRT for subtitle burning (default: <video_stem>.srt)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Skip highlights below this confidence (default: 0.0)",
    )
    parser.add_argument(
        "--no-burn",
        action="store_true",
        help="Skip subtitle burning",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: highlight/ next to video)",
    )
    parser.add_argument(
        "--min-dur",
        type=int,
        default=10,
        help="Minimum clip duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-dur",
        type=int,
        default=60,
        help="Maximum clip duration in seconds (default: 60)",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else video_path.parent / "highlight"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_srt = Path(args.srt).resolve() if args.srt else video_path.with_suffix(".srt")
    burn_enabled = not args.no_burn

    print(f"\n{'=' * 60}")
    print("  Post-processor")
    print(f"  Video      : {video_path.name}")
    print(f"  Highlights : {Path(args.highlights_md).name}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Burn-in    : {'No' if not burn_enabled else f'Yes ({resolved_srt.name})'}")
    print(f"{'=' * 60}\n")

    print("[1/3] Parsing highlight markdown...")
    all_highlights = parse_highlight_md(args.highlights_md)
    highlights = [h for h in all_highlights if h["confidence"] >= args.min_confidence]
    print(f"  {len(all_highlights)} highlights parsed, {len(highlights)} above confidence {args.min_confidence}")

    if not highlights:
        print("  No highlights to process. Exiting.")
        return

    print("\n[2/3] Reading video metadata...")
    video_duration = get_video_duration(str(video_path))
    print(f"  Duration: {video_duration/60:.1f} min")

    segments = None
    if burn_enabled:
        if not resolved_srt.exists():
            raise SystemExit(
                f"ERROR: subtitle burn-in requires an SRT file. Expected: {resolved_srt}"
            )
        segments = _parse_srt(str(resolved_srt))
        print(f"  Loaded SRT: {resolved_srt.name} ({len(segments)} segments)")

    print(f"\n[3/3] Cutting {len(highlights)} clips...")
    for i, highlight in enumerate(highlights, 1):
        start, end = clamp_duration(
            highlight["start_sec"],
            highlight["end_sec"],
            args.min_dur,
            args.max_dur,
            video_duration,
        )
        duration = end - start
        name = make_clip_name(i, start, highlight["reason"])
        clip_mp4 = str(out_dir / f"{name}.mp4")
        raw_mp4 = str(out_dir / f"_raw_{name}.mp4")
        clip_srt = str(out_dir / f"{name}.srt")

        if segments is not None:
            write_clip_srt(segments, start, end, clip_srt)

        if not burn_enabled:
            cut_clip(str(video_path), start, duration, clip_mp4)
        else:
            cut_clip(str(video_path), start, duration, raw_mp4)
            try:
                burn_subtitles(raw_mp4, clip_srt, clip_mp4)
            except subprocess.CalledProcessError as exc:
                if os.path.exists(raw_mp4):
                    os.remove(raw_mp4)
                raise RuntimeError(
                    f"Subtitle burn failed for clip {i:02d}; no unburned fallback was saved."
                ) from exc
            else:
                os.remove(raw_mp4)

        mm = int(start) // 60
        ss = int(start) % 60
        print(
            f"  [{i:02d}] {mm:02d}:{ss:02d}  {duration:.0f}s  "
            f"conf={highlight['confidence']:.2f}  -> {name}.mp4"
        )

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(highlights)} clips in: {out_dir}")
    print(f"{'=' * 60}\n")


def _parse_srt(srt_path: str) -> list:
    """Parse an SRT file into a list of segment dicts (start, end, text)."""
    text = Path(srt_path).read_text(encoding="utf-8")
    segments = []
    for block in re.split(r"\n\n+", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        time_line = lines[1]
        match = re.match(
            r"(\d+):(\d+):(\d+),(\d+)\s+-->\s+(\d+):(\d+):(\d+),(\d+)",
            time_line,
        )
        if not match:
            continue
        start = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
            + int(match.group(4)) / 1000
        )
        end = (
            int(match.group(5)) * 3600
            + int(match.group(6)) * 60
            + int(match.group(7))
            + int(match.group(8)) / 1000
        )
        text_value = " ".join(lines[2:]).strip()
        segments.append({"start": start, "end": end, "text": text_value})
    return segments


if __name__ == "__main__":
    main()
