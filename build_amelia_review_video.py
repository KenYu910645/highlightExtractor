#!/usr/bin/env python3
"""
Create a concatenated Amelia review video from raw detector windows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from amelia_event import select_top_windows_for_duration


def format_stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}m{int(seconds % 60):02d}s"


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def build_review_video(
    video_path: str | Path,
    detector_json: str | Path,
    *,
    target_fraction: float = 0.10,
    max_clip_sec: float = 5.0,
    out_path: str | Path | None = None,
) -> dict:
    video_path = Path(video_path).resolve()
    detector_path = Path(detector_json).resolve()
    payload = json.loads(detector_path.read_text(encoding="utf-8"))
    windows = payload.get("windows", [])
    video_duration = probe_duration(video_path)
    target_duration_sec = video_duration * target_fraction
    selected, dynamic_threshold = select_top_windows_for_duration(
        windows,
        target_duration_sec=target_duration_sec,
        max_clip_sec=max_clip_sec,
    )

    if not selected:
        raise SystemExit(
            f"No detector windows were selected from {detector_path.name}"
        )

    out_path = (
        Path(out_path).resolve()
        if out_path
        else video_path.parent / f"{video_path.stem}_amelia_ranked_review.mp4"
    )
    manifest_path = out_path.parent / f"{out_path.stem}_windows.json"
    manifest = []
    with tempfile.TemporaryDirectory(prefix=f"{out_path.stem}_", dir=str(out_path.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        clips_dir = temp_root / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        concat_path = temp_root / "concat.txt"

        concat_lines = []
        for index, item in enumerate(selected, start=1):
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
            duration = max(0.1, end_sec - start_sec)
            clip_name = (
                f"{out_path.stem}_{index:02d}_{format_stamp(start_sec)}_{format_stamp(end_sec)}.mp4"
            )
            clip_path = clips_dir / clip_name
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_sec:.3f}",
                    "-i",
                    str(video_path),
                    "-t",
                    f"{duration:.3f}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    str(clip_path),
                ],
                check=True,
                capture_output=True,
            )
            concat_lines.append(f"file '{clip_path.as_posix()}'")
            manifest.append(
                {
                    "index": index,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(duration, 3),
                    "center_sec": item["center_sec"],
                    "score": item["score"],
                }
            )

        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )

    manifest_path.write_text(
        json.dumps(
            {
                "dynamic_threshold": dynamic_threshold,
                "max_clip_sec": max_clip_sec,
                "target_fraction": target_fraction,
                "target_duration_sec": round(target_duration_sec, 3),
                "selected_duration_sec": round(sum(item["duration_sec"] for item in manifest), 3),
                "clip_count": len(manifest),
                "clips": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "review_video": str(out_path),
        "manifest": str(manifest_path),
        "clip_count": len(manifest),
        "target_duration_sec": round(target_duration_sec, 3),
        "dynamic_threshold": dynamic_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a concatenated Amelia review video from detector windows."
    )
    parser.add_argument("video", help="Source video path")
    parser.add_argument("detector_json", help="Path to *_amelia_events.json")
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.10,
        help="Target fraction of source runtime to keep (default: 0.10)",
    )
    parser.add_argument(
        "--max-clip-sec",
        type=float,
        default=5.0,
        help="Maximum length for each output clip (default: 5.0)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output review video path (default: <video dir>/<stem>_amelia_ranked_review.mp4)",
    )
    args = parser.parse_args()

    result = build_review_video(
        args.video,
        args.detector_json,
        target_fraction=args.target_fraction,
        max_clip_sec=args.max_clip_sec,
        out_path=args.out,
    )

    print(f"Review video : {result['review_video']}")
    print(f"Clip count   : {result['clip_count']}")
    print(f"Target secs  : {result['target_duration_sec']:.3f}")
    print(f"Threshold    : {result['dynamic_threshold']:g}")
    print(f"Manifest     : {result['manifest']}")


if __name__ == "__main__":
    main()
