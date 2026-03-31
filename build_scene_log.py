#!/usr/bin/env python3
"""
build_scene_log.py - Build a coarse semantic gameplay timeline with Qwen2.5-VL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from qwen_vl import (
    DEFAULT_MODEL,
    build_messages,
    extract_json_candidate,
    generate_answer,
    load_model_and_processor,
    path_to_file_uri,
    suppress_runtime_noise,
)


DEFAULT_FPS = 0.5
DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_SCENE_SUMMARY = "unclear"
DEFAULT_SSIM_THRESHOLD = 0.08
DEFAULT_MAX_SKIP_SEC = 8.0
DEFAULT_SSIM_SIZE = (64, 36)
SCENE_TYPES = {"menu", "traversal", "combat", "boss_combat", "cutscene", "death", "victory", "other"}
KEN_EXPRESSIONS = {"neutral", "tense", "surprised", "happy", "unclear"}
AMELIA_EXPRESSIONS = {"neutral", "tense", "surprised", "happy", "excited", "unclear"}
SCENE_PROMPT = """You are analyzing a gameplay video frame.

Return ONLY valid JSON.

Use only visible evidence. If unsure, use "unclear".

Fields:
- scene_type: one of [menu, traversal, combat, boss_combat, cutscene, death, victory, other]
- scene_summary: short phrase (max 10 words)
- important_entities: list of visible entities (e.g., boss, player, UI, facecam)
- intensity: integer 0-5
- danger_level: integer 0-5
- ken_expression: one of [neutral, tense, surprised, happy, unclear]
- amelia_expression: one of [neutral, tense, surprised, happy, excited, unclear]
- notable_change: true or false
- confidence: float between 0 and 1
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a coarse semantic scene timeline for a video using Qwen2.5-VL"
    )
    parser.add_argument("video", help="Input MP4 file path")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame sampling rate (default: 0.5)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens per frame (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional max pixel budget for Qwen")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for debug runs")
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=DEFAULT_SSIM_THRESHOLD,
        help=f"Minimum change score required to keep a frame for Qwen (default: {DEFAULT_SSIM_THRESHOLD})",
    )
    parser.add_argument(
        "--max-skip-sec",
        type=float,
        default=DEFAULT_MAX_SKIP_SEC,
        help=f"Maximum seconds to skip before forcing a fresh Qwen read (default: {DEFAULT_MAX_SKIP_SEC})",
    )
    parser.add_argument(
        "--ssim-size",
        default=f"{DEFAULT_SSIM_SIZE[0]}x{DEFAULT_SSIM_SIZE[1]}",
        help=f"Downscaled SSIM size WxH (default: {DEFAULT_SSIM_SIZE[0]}x{DEFAULT_SSIM_SIZE[1]})",
    )
    parser.add_argument(
        "--disable-ssim-gating",
        action="store_true",
        help="Send every sampled frame to Qwen and bypass SSIM-based skipping",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Temporarily required flag to enable Qwen-based scene logging",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract frames and re-run all frame analysis")
    return parser


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
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def get_output_paths(video_path: Path) -> dict[str, Path]:
    stem = video_path.stem
    base_dir = video_path.parent
    return {
        "frames_dir": base_dir / f"{stem}_scene_frames",
        "frame_index": base_dir / f"{stem}_scene_frame_index.json",
        "selection": base_dir / f"{stem}_scene_selection.json",
        "raw_log": base_dir / f"{stem}_scene_log_raw.json",
        "smoothed_log": base_dir / f"{stem}_scene_log_smoothed.json",
        "segments": base_dir / f"{stem}_scene_segments.json",
        "events": base_dir / f"{stem}_candidate_events.json",
    }


def clear_frame_dir(frames_dir: Path) -> None:
    if not frames_dir.exists():
        return
    for frame_path in frames_dir.glob("frame_*.jpg"):
        frame_path.unlink()


def extract_frames(video_path: Path, frames_dir: Path, fps: float, max_frames: int | None, force: bool) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing and not force:
        return existing[:max_frames] if max_frames is not None else existing

    clear_frame_dir(frames_dir)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "3",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(frames_dir / "frame_%06d.jpg"))
    subprocess.run(command, check=True, capture_output=True)
    return sorted(frames_dir.glob("frame_*.jpg"))


def build_frame_index(frame_paths: list[Path], fps: float, video_duration: float) -> list[dict[str, Any]]:
    interval = 1.0 / fps
    index: list[dict[str, Any]] = []
    for position, frame_path in enumerate(sorted(frame_paths), start=1):
        timestamp = round((position - 1) * interval, 3)
        index.append(
            {
                "frame": frame_path.name,
                "frame_path": str(frame_path.resolve()),
                "timestamp": min(timestamp, round(video_duration, 3)),
            }
        )
    return index


def load_json_file(path: Path, default: Any):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("--ssim-size must look like 64x36") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--ssim-size dimensions must be greater than 0")
    return width, height


def clamp_int(value: Any, low: int = 0, high: int = 5) -> int:
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def clamp_float(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def normalize_summary(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return DEFAULT_SCENE_SUMMARY
    words = text.split()
    return " ".join(words[:10])


def normalize_entities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = " ".join(str(item or "").strip().split())
        if not text:
            continue
        normalized = text[:40]
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(normalized)
    return cleaned


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def default_analysis() -> dict[str, Any]:
    return {
        "scene_type": "other",
        "scene_summary": DEFAULT_SCENE_SUMMARY,
        "important_entities": [],
        "intensity": 0,
        "danger_level": 0,
        "ken_expression": "unclear",
        "amelia_expression": "unclear",
        "notable_change": False,
        "confidence": 0.0,
    }


def parse_scene_analysis(raw_output: str) -> tuple[dict[str, Any], bool, str | None]:
    candidate = extract_json_candidate(raw_output.strip())
    if candidate is None:
        return default_analysis(), False, "no_json_object_found"

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return default_analysis(), False, "invalid_json"

    if not isinstance(payload, dict):
        return default_analysis(), False, "json_not_object"

    normalized = {
        "scene_type": normalize_choice(payload.get("scene_type"), SCENE_TYPES, "other"),
        "scene_summary": normalize_summary(payload.get("scene_summary")),
        "important_entities": normalize_entities(payload.get("important_entities")),
        "intensity": clamp_int(payload.get("intensity")),
        "danger_level": clamp_int(payload.get("danger_level")),
        "ken_expression": normalize_choice(payload.get("ken_expression"), KEN_EXPRESSIONS, "unclear"),
        "amelia_expression": normalize_choice(payload.get("amelia_expression"), AMELIA_EXPRESSIONS, "unclear"),
        "notable_change": normalize_bool(payload.get("notable_change")),
        "confidence": round(clamp_float(payload.get("confidence")), 4),
    }

    return normalized, True, None


def build_scene_messages(image_path: Path) -> list[dict[str, Any]]:
    return build_messages(path_to_file_uri(image_path), SCENE_PROMPT)


def analyze_frame(model: Any, processor: Any, frame_path: Path, max_new_tokens: int) -> tuple[dict[str, Any], bool, str, str | None]:
    raw_output = generate_answer(
        model=model,
        processor=processor,
        messages=build_scene_messages(frame_path),
        image_path=frame_path,
        max_new_tokens=max_new_tokens,
    )
    analysis, parse_ok, parse_error = parse_scene_analysis(raw_output)
    return analysis, parse_ok, raw_output.strip(), parse_error


def make_raw_entry(index_item: dict[str, Any], analysis: dict[str, Any], parse_ok: bool, raw_output: str, parse_error: str | None) -> dict[str, Any]:
    entry = {
        "timestamp": index_item["timestamp"],
        "frame": index_item["frame"],
        "analysis": analysis,
        "parse_ok": parse_ok,
    }
    if not parse_ok:
        entry["raw_output"] = raw_output
        entry["parse_error"] = parse_error
    return entry


def load_grayscale_frame(frame_path: Path, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.open(frame_path).convert("L").resize(size)
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_ssim_score(reference: np.ndarray, current: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(reference, current, data_range=1.0))


def select_frames_for_qwen(
    frame_index: list[dict[str, Any]],
    *,
    ssim_threshold: float,
    max_skip_sec: float,
    ssim_size: tuple[int, int],
    disable_ssim_gating: bool,
) -> list[dict[str, Any]]:
    selection: list[dict[str, Any]] = []
    if not frame_index:
        return selection

    last_selected_index_item: dict[str, Any] | None = None
    last_selected_gray: np.ndarray | None = None

    for idx, item in enumerate(frame_index):
        if idx == 0:
            last_selected_index_item = item
            last_selected_gray = load_grayscale_frame(Path(item["frame_path"]), ssim_size)
            selection.append(
                {
                    "timestamp": item["timestamp"],
                    "frame": item["frame"],
                    "ssim": None,
                    "change_score": None,
                    "selected_for_qwen": True,
                    "selection_reason": "first_frame",
                }
            )
            continue

        current_gray = load_grayscale_frame(Path(item["frame_path"]), ssim_size)
        ssim = compute_ssim_score(last_selected_gray, current_gray)
        change_score = round(1.0 - ssim, 6)
        seconds_since_last_selected = item["timestamp"] - last_selected_index_item["timestamp"]

        if disable_ssim_gating:
            selected_for_qwen = True
            selection_reason = "gating_disabled"
        elif change_score >= ssim_threshold:
            selected_for_qwen = True
            selection_reason = "ssim_change"
        elif seconds_since_last_selected >= max_skip_sec:
            selected_for_qwen = True
            selection_reason = "max_skip_sec"
        else:
            selected_for_qwen = False
            selection_reason = "propagated"

        selection.append(
            {
                "timestamp": item["timestamp"],
                "frame": item["frame"],
                "ssim": round(ssim, 6),
                "change_score": change_score,
                "selected_for_qwen": selected_for_qwen,
                "selection_reason": selection_reason,
            }
        )

        if selected_for_qwen:
            last_selected_index_item = item
            last_selected_gray = current_gray

    return selection


def build_raw_log_with_propagation(
    frame_index: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    qwen_entries_by_key: dict[tuple[str, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_log: list[dict[str, Any]] = []
    latest_selected_entry: dict[str, Any] | None = None
    selection_map = {
        (item["frame"], float(item["timestamp"])): item
        for item in selection
    }

    for item in frame_index:
        key = (item["frame"], float(item["timestamp"]))
        selection_item = selection_map[key]
        if selection_item["selected_for_qwen"]:
            entry = dict(qwen_entries_by_key[key])
            entry["analysis_source"] = "qwen"
            latest_selected_entry = entry
        else:
            entry = {
                "timestamp": item["timestamp"],
                "frame": item["frame"],
                "analysis": dict(latest_selected_entry["analysis"]),
                "parse_ok": latest_selected_entry["parse_ok"],
                "analysis_source": "propagated",
                "propagated_from_timestamp": latest_selected_entry["timestamp"],
            }
            if "raw_output" in latest_selected_entry:
                entry["raw_output"] = latest_selected_entry["raw_output"]
            if "parse_error" in latest_selected_entry:
                entry["parse_error"] = latest_selected_entry["parse_error"]

        entry["selected_for_qwen"] = selection_item["selected_for_qwen"]
        entry["selection_reason"] = selection_item["selection_reason"]
        entry["ssim"] = selection_item["ssim"]
        entry["change_score"] = selection_item["change_score"]
        raw_log.append(entry)

    return raw_log


def analyze_frames(
    frame_index: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    raw_log_path: Path,
    model_name: str,
    max_pixels: int | None,
    max_new_tokens: int,
    force: bool,
) -> list[dict[str, Any]]:
    cached_entries = load_json_file(raw_log_path, [])
    cached_map = {
        (item.get("frame"), float(item.get("timestamp", -1))): item
        for item in cached_entries
        if isinstance(item, dict) and item.get("analysis_source") == "qwen"
    }
    qwen_results: dict[tuple[str, float], dict[str, Any]] = {}
    pending = []
    selected_map = {
        (item["frame"], float(item["timestamp"])): item
        for item in selection
        if item["selected_for_qwen"]
    }

    for item in frame_index:
        key = (item["frame"], float(item["timestamp"]))
        if key not in selected_map:
            continue
        if not force and key in cached_map:
            qwen_results[key] = cached_map[key]
        else:
            pending.append(item)

    if pending:
        with suppress_runtime_noise():
            model, processor = load_model_and_processor(model_name, max_pixels)
            for item in pending:
                analysis, parse_ok, raw_output, parse_error = analyze_frame(
                    model=model,
                    processor=processor,
                    frame_path=Path(item["frame_path"]),
                    max_new_tokens=max_new_tokens,
                )
                key = (item["frame"], float(item["timestamp"]))
                qwen_results[key] = make_raw_entry(item, analysis, parse_ok, raw_output, parse_error)

    return build_raw_log_with_propagation(frame_index, selection, qwen_results)


def smooth_numeric(values: list[int]) -> list[int]:
    smoothed: list[int] = []
    for idx in range(len(values)):
        lo = max(0, idx - 1)
        hi = min(len(values), idx + 2)
        avg = sum(values[lo:hi]) / len(values[lo:hi])
        smoothed.append(max(0, min(5, int(round(avg)))))
    return smoothed


def smooth_expression_sequence(values: list[str], confidences: list[float], high_confidence: float = 0.85) -> list[str]:
    smoothed = list(values)
    for idx in range(1, len(values) - 1):
        if values[idx - 1] == values[idx + 1] and values[idx] != values[idx - 1]:
            if confidences[idx] < high_confidence:
                smoothed[idx] = values[idx - 1]
    return smoothed


def smooth_scene_types(entries: list[dict[str, Any]]) -> list[str]:
    scene_types = [entry["analysis"]["scene_type"] for entry in entries]
    confidences = [entry["analysis"]["confidence"] for entry in entries]
    smoothed = list(scene_types)
    for idx in range(1, len(scene_types) - 1):
        left = scene_types[idx - 1]
        center = scene_types[idx]
        right = scene_types[idx + 1]
        if left != right:
            continue
        center_conf = confidences[idx]
        neighbor_conf = max(confidences[idx - 1], confidences[idx + 1])
        if center_conf >= neighbor_conf + 0.2:
            continue
        smoothed[idx] = left
    return smoothed


def smooth_scene_log(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_entries:
        return []

    scene_types = smooth_scene_types(raw_entries)
    intensities = smooth_numeric([entry["analysis"]["intensity"] for entry in raw_entries])
    danger_levels = smooth_numeric([entry["analysis"]["danger_level"] for entry in raw_entries])
    confidences = [entry["analysis"]["confidence"] for entry in raw_entries]
    ken_values = smooth_expression_sequence([entry["analysis"]["ken_expression"] for entry in raw_entries], confidences)
    amelia_values = smooth_expression_sequence(
        [entry["analysis"]["amelia_expression"] for entry in raw_entries],
        confidences,
    )

    smoothed_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        analysis = dict(entry["analysis"])
        analysis["scene_type"] = scene_types[idx]
        analysis["intensity"] = intensities[idx]
        analysis["danger_level"] = danger_levels[idx]
        analysis["ken_expression"] = ken_values[idx]
        analysis["amelia_expression"] = amelia_values[idx]
        smoothed_entry = {
            "timestamp": entry["timestamp"],
            "frame": entry["frame"],
            "parse_ok": entry["parse_ok"],
            "raw_analysis": entry["analysis"],
            "analysis": analysis,
            "analysis_source": entry.get("analysis_source", "qwen"),
            "selected_for_qwen": entry.get("selected_for_qwen", True),
            "selection_reason": entry.get("selection_reason"),
            "ssim": entry.get("ssim"),
            "change_score": entry.get("change_score"),
        }
        if "propagated_from_timestamp" in entry:
            smoothed_entry["propagated_from_timestamp"] = entry["propagated_from_timestamp"]
        if "raw_output" in entry:
            smoothed_entry["raw_output"] = entry["raw_output"]
        if "parse_error" in entry:
            smoothed_entry["parse_error"] = entry["parse_error"]
        smoothed_entries.append(smoothed_entry)

    return smoothed_entries


def dominant_value(values: list[str], default: str) -> str:
    values = [value for value in values if value]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def merge_scene_segments(smoothed_entries: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    if not smoothed_entries:
        return []

    interval = 1.0 / fps
    segments: list[dict[str, Any]] = []
    start_idx = 0

    def build_segment(chunk: list[dict[str, Any]], next_timestamp: float) -> dict[str, Any]:
        first = chunk[0]
        analyses = [item["analysis"] for item in chunk]
        best_summary = max(analyses, key=lambda item: item["confidence"])["scene_summary"]
        entity_counts = Counter(entity for item in analyses for entity in item["important_entities"])
        return {
            "start": first["timestamp"],
            "end": round(next_timestamp, 3),
            "scene_type": dominant_value([item["scene_type"] for item in analyses], "other"),
            "avg_intensity": round(sum(item["intensity"] for item in analyses) / len(analyses), 3),
            "avg_danger_level": round(sum(item["danger_level"] for item in analyses) / len(analyses), 3),
            "ken_expression": dominant_value([item["ken_expression"] for item in analyses], "unclear"),
            "amelia_expression": dominant_value([item["amelia_expression"] for item in analyses], "unclear"),
            "scene_summary": best_summary,
            "important_entities": [item for item, _ in entity_counts.most_common()],
            "avg_confidence": round(sum(item["confidence"] for item in analyses) / len(analyses), 4),
            "frame_count": len(chunk),
        }

    for idx in range(1, len(smoothed_entries) + 1):
        if idx == len(smoothed_entries):
            next_timestamp = smoothed_entries[-1]["timestamp"] + interval
            segments.append(build_segment(smoothed_entries[start_idx:idx], next_timestamp))
            break

        previous = smoothed_entries[idx - 1]["analysis"]["scene_type"]
        current = smoothed_entries[idx]["analysis"]["scene_type"]
        gap = smoothed_entries[idx]["timestamp"] - smoothed_entries[idx - 1]["timestamp"]
        if current == previous and gap <= interval * 1.5:
            continue

        segments.append(build_segment(smoothed_entries[start_idx:idx], smoothed_entries[idx]["timestamp"]))
        start_idx = idx

    return segments


def extract_candidate_events(smoothed_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, entry in enumerate(smoothed_entries):
        analysis = entry["analysis"]
        timestamp = entry["timestamp"]
        confidence = analysis["confidence"]
        if idx == 0:
            if analysis["notable_change"]:
                events.append(
                    {
                        "timestamp": timestamp,
                        "event_type": "notable_change",
                        "reason": "frame marked notable_change",
                        "confidence": confidence,
                    }
                )
            continue

        previous = smoothed_entries[idx - 1]["analysis"]
        if analysis["scene_type"] != previous["scene_type"]:
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": "scene_transition",
                    "reason": f"{previous['scene_type']} -> {analysis['scene_type']}",
                    "confidence": max(previous["confidence"], confidence),
                }
            )
        if analysis["intensity"] - previous["intensity"] >= 2:
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": "intensity_spike",
                    "reason": f"intensity {previous['intensity']} -> {analysis['intensity']}",
                    "confidence": confidence,
                }
            )
        if analysis["danger_level"] - previous["danger_level"] >= 2:
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": "danger_spike",
                    "reason": f"danger {previous['danger_level']} -> {analysis['danger_level']}",
                    "confidence": confidence,
                }
            )
        if analysis["ken_expression"] != previous["ken_expression"] and analysis["ken_expression"] not in {"neutral", "unclear"}:
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": "ken_expression_change",
                    "reason": f"Ken {previous['ken_expression']} -> {analysis['ken_expression']}",
                    "confidence": confidence,
                }
            )
        if analysis["amelia_expression"] != previous["amelia_expression"] and analysis["amelia_expression"] not in {"neutral", "unclear"}:
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": "amelia_expression_change",
                    "reason": f"Amelia {previous['amelia_expression']} -> {analysis['amelia_expression']}",
                    "confidence": confidence,
                }
            )
        if analysis["notable_change"]:
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": "notable_change",
                    "reason": "frame marked notable_change",
                    "confidence": confidence,
                }
            )
    return events


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ssim_size = parse_size(args.ssim_size)

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than 0")
    if args.max_pixels is not None and args.max_pixels <= 0:
        parser.error("--max-pixels must be greater than 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than 0")
    if args.ssim_threshold < 0 or args.ssim_threshold > 1:
        parser.error("--ssim-threshold must be between 0 and 1")
    if args.max_skip_sec <= 0:
        parser.error("--max-skip-sec must be greater than 0")
    if not args.enable_vlm:
        print(
            "VLM scene logging is temporarily disabled. Re-run with --enable-vlm "
            "when you want to use the Qwen-based pipeline again.",
            file=sys.stderr,
        )
        return 1

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        return 1

    outputs = get_output_paths(video_path)
    video_duration = probe_duration(video_path)

    print(f"\n{'=' * 60}")
    print("  Scene Timeline Builder")
    print(f"  Video      : {video_path.name}")
    print(f"  FPS        : {args.fps}")
    print(f"  Max frames : {args.max_frames if args.max_frames is not None else 'all'}")
    print(f"  SSIM gate  : {'off' if args.disable_ssim_gating else f'{args.ssim_threshold:.2f} @ {ssim_size[0]}x{ssim_size[1]}'}")
    print(f"{'=' * 60}\n")

    print("[1/5] Sampling frames...")
    frame_paths = extract_frames(video_path, outputs["frames_dir"], args.fps, args.max_frames, args.force)
    frame_index = build_frame_index(frame_paths, args.fps, video_duration)
    write_json(outputs["frame_index"], frame_index)
    print(f"  Frames sampled: {len(frame_index)}")

    print("[2/5] Selecting frames for Qwen...")
    selection = select_frames_for_qwen(
        frame_index,
        ssim_threshold=args.ssim_threshold,
        max_skip_sec=args.max_skip_sec,
        ssim_size=ssim_size,
        disable_ssim_gating=args.disable_ssim_gating,
    )
    write_json(outputs["selection"], selection)
    selected_count = sum(1 for item in selection if item["selected_for_qwen"])
    print(f"  Frames sent to Qwen: {selected_count}/{len(selection)}")

    print("[3/5] Running Qwen frame analysis...")
    raw_log = analyze_frames(
        frame_index=frame_index,
        selection=selection,
        raw_log_path=outputs["raw_log"],
        model_name=args.model,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        force=args.force,
    )
    write_json(outputs["raw_log"], raw_log)
    print(f"  Raw entries: {len(raw_log)}")

    print("[4/5] Smoothing and segmenting timeline...")
    smoothed_log = smooth_scene_log(raw_log)
    segments = merge_scene_segments(smoothed_log, args.fps)
    write_json(outputs["smoothed_log"], smoothed_log)
    write_json(outputs["segments"], segments)
    print(f"  Segments   : {len(segments)}")

    print("[5/5] Extracting candidate events...")
    events = extract_candidate_events(smoothed_log)
    write_json(outputs["events"], events)
    print(f"  Events     : {len(events)}")

    print("\nOutputs:")
    for key in ("frame_index", "selection", "raw_log", "smoothed_log", "segments", "events"):
        print(f"  - {outputs[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
