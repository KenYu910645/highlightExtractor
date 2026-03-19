#!/usr/bin/env python3
"""
Prepare reusable speaker enrollment clips from raw media.

This script converts MP4/audio enrollment files into cleaned mono 16 kHz WAV
clips under data/enroll/<speaker>/prepared/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from speaker_labeler.audio import (
    decode_audio_to_array,
    estimate_voiced_seconds,
    is_supported_source_file,
    resolve_media_binary,
    write_wav_file,
)


@dataclass
class PrepConfig:
    input_dir: Path
    speaker: str | None
    ffmpeg: str | None
    ffprobe: str | None
    sample_rate: int
    min_clip_sec: float
    max_clip_sec: float
    min_voice_sec: float
    frame_ms: int
    hop_ms: int
    energy_threshold: float
    max_silence_sec: float
    edge_padding_sec: float
    dry_run: bool


def frame_rms(samples: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    values = []
    for start in range(0, max(1, len(samples) - frame_len + 1), hop_len):
        frame = samples[start:start + frame_len]
        if len(frame) < frame_len:
            break
        values.append(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
    return np.asarray(values, dtype=np.float32)


def detect_voiced_intervals(samples: np.ndarray, sample_rate: int, config: PrepConfig) -> list[tuple[float, float]]:
    if len(samples) == 0:
        return []

    frame_len = max(1, int(sample_rate * config.frame_ms / 1000))
    hop_len = max(1, int(sample_rate * config.hop_ms / 1000))
    rms_values = frame_rms(samples, frame_len, hop_len)
    if len(rms_values) == 0:
        return []

    baseline = float(np.percentile(rms_values, 35))
    threshold = max(config.energy_threshold, baseline * 1.8)
    voiced = rms_values >= threshold

    intervals = []
    start_frame = None
    for index, is_voiced in enumerate(voiced):
        if is_voiced and start_frame is None:
            start_frame = index
        elif not is_voiced and start_frame is not None:
            intervals.append((start_frame * hop_len / sample_rate, index * hop_len / sample_rate))
            start_frame = None
    if start_frame is not None:
        intervals.append((start_frame * hop_len / sample_rate, len(voiced) * hop_len / sample_rate))

    merged = merge_intervals(intervals, config.max_silence_sec)
    return trim_and_filter_intervals(samples, sample_rate, merged, config)


def merge_intervals(intervals: list[tuple[float, float]], max_gap_sec: float) -> list[tuple[float, float]]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_sec:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def trim_and_filter_intervals(
    samples: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    config: PrepConfig,
) -> list[tuple[float, float]]:
    cleaned = []
    total_duration = len(samples) / sample_rate
    for start, end in intervals:
        start = max(0.0, start - config.edge_padding_sec)
        end = min(total_duration, end + config.edge_padding_sec)
        if end - start < config.min_clip_sec * 0.5:
            continue
        clip = samples[int(start * sample_rate):int(end * sample_rate)]
        voice_sec = estimate_voiced_seconds(clip, sample_rate, config.frame_ms, config.hop_ms, config.energy_threshold)
        if voice_sec < config.min_voice_sec:
            continue
        cleaned.append((start, end))
    return cleaned


def split_long_interval(
    samples: np.ndarray,
    sample_rate: int,
    interval: tuple[float, float],
    config: PrepConfig,
) -> list[tuple[float, float]]:
    start, end = interval
    duration = end - start
    if duration <= config.max_clip_sec:
        return [interval]

    pieces = []
    current_start = start
    while end - current_start > config.max_clip_sec:
        target_end = current_start + config.max_clip_sec
        split_point = find_split_point(samples, sample_rate, current_start, target_end, config)
        if split_point - current_start < config.min_clip_sec:
            split_point = min(end, current_start + config.max_clip_sec)
        pieces.append((current_start, split_point))
        current_start = split_point

    if end - current_start >= config.min_clip_sec:
        pieces.append((current_start, end))
    elif pieces:
        last_start, _ = pieces[-1]
        pieces[-1] = (last_start, end)
    else:
        pieces.append((start, end))
    return pieces


def find_split_point(
    samples: np.ndarray,
    sample_rate: int,
    start_sec: float,
    target_end_sec: float,
    config: PrepConfig,
) -> float:
    frame_len = max(1, int(sample_rate * config.frame_ms / 1000))
    hop_len = max(1, int(sample_rate * config.hop_ms / 1000))
    region_start = int(max(0.0, (target_end_sec - 0.8) * sample_rate))
    region_end = int(min(len(samples), (target_end_sec + 0.8) * sample_rate))
    region = samples[region_start:region_end]
    if len(region) < frame_len:
        return target_end_sec

    rms_values = frame_rms(region, frame_len, hop_len)
    if len(rms_values) == 0:
        return target_end_sec
    quiet_index = int(np.argmin(rms_values))
    split_sec = (region_start + quiet_index * hop_len) / sample_rate
    return max(start_sec, split_sec)


def process_source(path: Path, prepared_dir: Path, config: PrepConfig) -> dict:
    samples, sample_rate = decode_audio_to_array(str(path), sample_rate=config.sample_rate)
    intervals = detect_voiced_intervals(samples, sample_rate, config)

    clip_records = []
    clip_index = 1
    for interval in intervals:
        for split_start, split_end in split_long_interval(samples, sample_rate, interval, config):
            clip = samples[int(split_start * sample_rate):int(split_end * sample_rate)]
            duration = len(clip) / sample_rate
            voice_sec = estimate_voiced_seconds(clip, sample_rate, config.frame_ms, config.hop_ms, config.energy_threshold)
            if duration < config.min_clip_sec or voice_sec < config.min_voice_sec:
                continue

            clip_name = f"{path.stem}_{clip_index:03d}.wav"
            output_path = prepared_dir / clip_name
            if not config.dry_run:
                write_wav_file(output_path, clip, sample_rate)
            clip_records.append({
                "output": str(output_path),
                "start_sec": round(split_start, 3),
                "end_sec": round(split_end, 3),
                "duration_sec": round(duration, 3),
                "voice_sec": round(voice_sec, 3),
            })
            clip_index += 1

    return {
        "source": str(path),
        "clips": clip_records,
        "clip_count": len(clip_records),
    }


def process_speaker(speaker_dir: Path, config: PrepConfig) -> dict:
    prepared_dir = speaker_dir / "prepared"
    source_files = [
        path for path in sorted(speaker_dir.iterdir())
        if is_supported_source_file(path) and path.parent == speaker_dir
    ]

    summary = {
        "speaker": speaker_dir.name,
        "source_count": len(source_files),
        "prepared_dir": str(prepared_dir),
        "sources": [],
        "total_clips": 0,
    }

    if not config.dry_run:
        prepared_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in prepared_dir.iterdir():
            if stale_file.is_file() and stale_file.suffix.lower() in {".wav", ".json", ".txt"}:
                stale_file.unlink()

    for path in source_files:
        record = process_source(path, prepared_dir, config)
        summary["sources"].append(record)
        summary["total_clips"] += record["clip_count"]

    manifest_path = prepared_dir / "manifest.json"
    if not config.dry_run:
        manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary


def speaker_dirs(input_dir: Path, selected_speaker: str | None) -> list[Path]:
    names = [selected_speaker] if selected_speaker else ["ken", "amelia"]
    return [input_dir / name for name in names]


def parse_args() -> PrepConfig:
    parser = argparse.ArgumentParser(description="Prepare enrollment WAV clips from raw media.")
    parser.add_argument("--input-dir", default="data/enroll", help="Enrollment root directory")
    parser.add_argument("--speaker", choices=["ken", "amelia"], default=None, help="Process only one speaker")
    parser.add_argument("--ffmpeg", default=None, help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default=None, help="Path to ffprobe executable")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate (default: 16000)")
    parser.add_argument("--min-clip-sec", type=float, default=1.5, help="Minimum kept clip duration")
    parser.add_argument("--max-clip-sec", type=float, default=6.0, help="Maximum target clip duration")
    parser.add_argument("--min-voice-sec", type=float, default=0.7, help="Minimum voiced speech per clip")
    parser.add_argument("--frame-ms", type=int, default=30, help="Analysis frame size in ms")
    parser.add_argument("--hop-ms", type=int, default=15, help="Analysis hop size in ms")
    parser.add_argument("--energy-threshold", type=float, default=0.015, help="Minimum RMS threshold for voiced frames")
    parser.add_argument("--max-silence-sec", type=float, default=0.35, help="Maximum silence gap to merge across")
    parser.add_argument("--edge-padding-sec", type=float, default=0.12, help="Padding added around voiced regions")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing WAVs")
    args = parser.parse_args()

    return PrepConfig(
        input_dir=Path(args.input_dir).resolve(),
        speaker=args.speaker,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        sample_rate=args.sample_rate,
        min_clip_sec=args.min_clip_sec,
        max_clip_sec=args.max_clip_sec,
        min_voice_sec=args.min_voice_sec,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        energy_threshold=args.energy_threshold,
        max_silence_sec=args.max_silence_sec,
        edge_padding_sec=args.edge_padding_sec,
        dry_run=args.dry_run,
    )


def validate_media_tools(config: PrepConfig) -> None:
    ffmpeg_path = resolve_media_binary("ffmpeg", config.ffmpeg)
    ffprobe_path = resolve_media_binary("ffprobe", config.ffprobe)
    if not Path(ffmpeg_path).exists() and ffmpeg_path == "ffmpeg":
        raise SystemExit("ffmpeg not found. Use --ffmpeg to point at the executable.")
    if not Path(ffprobe_path).exists() and ffprobe_path == "ffprobe":
        raise SystemExit("ffprobe not found. Use --ffprobe to point at the executable.")
    if Path(ffmpeg_path).exists():
        import os
        os.environ["FFMPEG_BIN"] = ffmpeg_path
    if Path(ffprobe_path).exists():
        import os
        os.environ["FFPROBE_BIN"] = ffprobe_path


def main() -> None:
    config = parse_args()
    validate_media_tools(config)

    print(f"\n{'=' * 60}")
    print("  Enrollment Prep")
    print(f"  Input dir   : {config.input_dir}")
    print(f"  Speaker     : {config.speaker or 'all'}")
    print(f"  Dry run     : {'Yes' if config.dry_run else 'No'}")
    print(f"  Clip window : {config.min_clip_sec:.1f}s to {config.max_clip_sec:.1f}s")
    print(f"{'=' * 60}\n")

    summaries = []
    for speaker_dir in speaker_dirs(config.input_dir, config.speaker):
        if not speaker_dir.exists():
            print(f"WARNING: speaker directory not found: {speaker_dir}")
            continue
        summary = process_speaker(speaker_dir, config)
        summaries.append(summary)

        print(f"[{speaker_dir.name}] {summary['source_count']} source files -> {summary['total_clips']} prepared clips")
        for source in summary["sources"]:
            print(f"  - {Path(source['source']).name}: {source['clip_count']} clips")
        if not config.dry_run:
            print(f"  - manifest: {summary['manifest']}")
        print("")

    if not summaries:
        raise SystemExit("No speaker directories were processed.")


if __name__ == "__main__":
    main()
