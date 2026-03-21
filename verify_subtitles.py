#!/usr/bin/env python3
"""Evaluate generated subtitles against manual ground truth."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class SubtitleEntry:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class SubtitleMatch:
    pred_index: int
    gt_index: int
    overlap_sec: float
    text_similarity: float
    start_error_sec: float
    end_error_sec: float


def parse_timestamp(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def normalize_text(text: str) -> str:
    """Normalize subtitle text for deterministic comparison."""
    collapsed = re.sub(r"\s+", "", text)
    return collapsed.strip()


def parse_srt(path: str | Path) -> list[SubtitleEntry]:
    """Parse an SRT file into normalized subtitle entries."""
    raw_text = Path(path).read_text(encoding="utf-8")
    entries: list[SubtitleEntry] = []

    for block in re.split(r"\n\s*\n+", raw_text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        time_line = lines[1]
        if "-->" not in time_line:
            continue

        start_str, end_str = [part.strip() for part in time_line.split("-->", 1)]
        try:
            start_sec = parse_timestamp(start_str)
            end_sec = parse_timestamp(end_str)
        except ValueError:
            continue

        text = normalize_text("".join(lines[2:]))
        if not text:
            continue

        end_sec = max(start_sec, end_sec)
        entries.append(SubtitleEntry(start_sec=start_sec, end_sec=end_sec, text=text))

    return entries


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein distance for character-level comparisons."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_cer(pred_entries: Iterable[SubtitleEntry], gt_entries: Iterable[SubtitleEntry]) -> float:
    """Compute global character error rate on concatenated subtitle text."""
    pred_text = "".join(entry.text for entry in pred_entries)
    gt_text = "".join(entry.text for entry in gt_entries)
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    return levenshtein_distance(pred_text, gt_text) / len(gt_text)


def text_similarity(left: str, right: str) -> float:
    """Return a character-level similarity ratio in [0, 1]."""
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(left, right) / max_len)


def compute_overlap_seconds(left: SubtitleEntry, right: SubtitleEntry) -> float:
    """Return temporal overlap between two subtitle intervals."""
    return max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))


def match_subtitles(
    pred_entries: list[SubtitleEntry],
    gt_entries: list[SubtitleEntry],
    *,
    search_tolerance_sec: float = 1.0,
) -> list[SubtitleMatch]:
    """
    Greedily match predicted subtitles to ground truth by overlap, then text similarity.

    Candidate pairs must overlap or have a close boundary within the search tolerance.
    """
    candidates: list[tuple[float, float, float, int, int]] = []

    for pred_index, pred in enumerate(pred_entries):
        for gt_index, gt in enumerate(gt_entries):
            overlap = compute_overlap_seconds(pred, gt)
            boundary_gap = min(
                abs(pred.start_sec - gt.start_sec),
                abs(pred.end_sec - gt.end_sec),
            )
            if overlap <= 0.0 and boundary_gap > search_tolerance_sec:
                continue

            similarity = text_similarity(pred.text, gt.text)
            candidates.append((overlap, similarity, -boundary_gap, pred_index, gt_index))

    candidates.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[SubtitleMatch] = []

    for overlap, similarity, _, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue

        pred = pred_entries[pred_index]
        gt = gt_entries[gt_index]
        matches.append(
            SubtitleMatch(
                pred_index=pred_index,
                gt_index=gt_index,
                overlap_sec=overlap,
                text_similarity=similarity,
                start_error_sec=abs(pred.start_sec - gt.start_sec),
                end_error_sec=abs(pred.end_sec - gt.end_sec),
            )
        )
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)

    matches.sort(key=lambda match: (match.pred_index, match.gt_index))
    return matches


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _safe_median(values: list[float]) -> float:
    return median(values) if values else math.nan


def format_seconds_ms(value: float) -> str:
    """Format seconds as milliseconds for report output."""
    if math.isnan(value):
        return "n/a"
    return f"{value * 1000:.1f} ms"


def evaluate_subtitles(
    pred_entries: list[SubtitleEntry],
    gt_entries: list[SubtitleEntry],
    *,
    text_similarity_threshold: float = 0.5,
    timing_tolerance_sec: float = 0.5,
    search_tolerance_sec: float = 1.0,
) -> dict:
    """Compute CER, timing metrics, and match precision/recall/F1."""
    matches = match_subtitles(
        pred_entries,
        gt_entries,
        search_tolerance_sec=search_tolerance_sec,
    )

    timing_matches = [
        match for match in matches
        if match.text_similarity >= text_similarity_threshold
    ]

    correct_matches = [
        match for match in timing_matches
        if match.start_error_sec <= timing_tolerance_sec
        and match.end_error_sec <= timing_tolerance_sec
    ]

    start_errors = [match.start_error_sec for match in timing_matches]
    end_errors = [match.end_error_sec for match in timing_matches]

    precision = len(correct_matches) / len(pred_entries) if pred_entries else 0.0
    recall = len(correct_matches) / len(gt_entries) if gt_entries else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "pred_count": len(pred_entries),
        "gt_count": len(gt_entries),
        "global_cer": compute_cer(pred_entries, gt_entries),
        "matched_pairs": len(timing_matches),
        "start_error_mean_sec": _safe_mean(start_errors),
        "start_error_median_sec": _safe_median(start_errors),
        "end_error_mean_sec": _safe_mean(end_errors),
        "end_error_median_sec": _safe_median(end_errors),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_matches": len(correct_matches),
        "timing_tolerance_sec": timing_tolerance_sec,
        "text_similarity_threshold": text_similarity_threshold,
    }


def build_report(pred_path: Path, gt_path: Path, metrics: dict) -> str:
    """Render a compact human-readable evaluation report."""
    lines = [
        "Subtitle Verification",
        f"Predicted file : {pred_path}",
        f"Ground truth   : {gt_path}",
        f"Pred blocks    : {metrics['pred_count']}",
        f"GT blocks      : {metrics['gt_count']}",
        f"Global CER     : {metrics['global_cer']:.4f}",
        (
            "Matched pairs  : "
            f"{metrics['matched_pairs']} "
            f"(text similarity >= {metrics['text_similarity_threshold']:.2f})"
        ),
        (
            "Start error    : "
            f"mean {format_seconds_ms(metrics['start_error_mean_sec'])}, "
            f"median {format_seconds_ms(metrics['start_error_median_sec'])}"
        ),
        (
            "End error      : "
            f"mean {format_seconds_ms(metrics['end_error_mean_sec'])}, "
            f"median {format_seconds_ms(metrics['end_error_median_sec'])}"
        ),
        (
            "Match F1       : "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"(<= {metrics['timing_tolerance_sec']:.1f}s timing tolerance)"
        ),
        f"Correct matches: {metrics['correct_matches']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated subtitles against manually labeled ground truth"
    )
    parser.add_argument("--pred", required=True, help="Path to predicted/generated SRT file")
    parser.add_argument("--gt", required=True, help="Path to ground-truth SRT file")
    parser.add_argument(
        "--timing-tolerance",
        type=float,
        default=0.5,
        help="Maximum allowed absolute start/end error in seconds for F1 correctness (default: 0.5)",
    )
    parser.add_argument(
        "--text-sim-threshold",
        type=float,
        default=0.5,
        help="Minimum text similarity for a matched pair to count toward timing metrics and F1 (default: 0.5)",
    )
    parser.add_argument(
        "--search-tolerance",
        type=float,
        default=1.0,
        help="Search window in seconds for candidate subtitle matches when there is no overlap (default: 1.0)",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred).resolve()
    gt_path = Path(args.gt).resolve()

    pred_entries = parse_srt(pred_path)
    gt_entries = parse_srt(gt_path)

    metrics = evaluate_subtitles(
        pred_entries,
        gt_entries,
        text_similarity_threshold=args.text_sim_threshold,
        timing_tolerance_sec=args.timing_tolerance,
        search_tolerance_sec=args.search_tolerance,
    )
    print(build_report(pred_path, gt_path, metrics))


if __name__ == "__main__":
    main()
