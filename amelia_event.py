#!/usr/bin/env python3
"""
Few-shot Amelia vocal event detection using pretrained speech embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WINDOW_SEC = 1.5
DEFAULT_HOP_SEC = 0.5
DEFAULT_SCORE_THRESHOLD = 0.58
DEFAULT_MAX_GAP_SEC = 0.75
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_SOURCE = "pretrained_models/speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PROTOTYPE_PATH = "data/enroll/amelia_event_prototypes.json"


@dataclass
class AmeliaEventConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    window_sec: float = DEFAULT_WINDOW_SEC
    hop_sec: float = DEFAULT_HOP_SEC
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class AmeliaEventDetector:
    prototype_path: Path
    model_source: str = DEFAULT_MODEL_SOURCE
    device: str | None = None

    def __post_init__(self) -> None:
        payload = json.loads(self.prototype_path.read_text(encoding="utf-8"))
        config_payload = payload.get("config", {})
        self.config = AmeliaEventConfig(
            sample_rate=int(config_payload.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            window_sec=float(config_payload.get("window_sec", DEFAULT_WINDOW_SEC)),
            hop_sec=float(config_payload.get("hop_sec", DEFAULT_HOP_SEC)),
            score_threshold=float(config_payload.get("score_threshold", DEFAULT_SCORE_THRESHOLD)),
            max_gap_sec=float(config_payload.get("max_gap_sec", DEFAULT_MAX_GAP_SEC)),
            batch_size=int(config_payload.get("batch_size", DEFAULT_BATCH_SIZE)),
        )
        self.model_source = payload.get("model_source", self.model_source)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prototype_embeddings = _normalize_rows(
            np.asarray(payload["prototype_embeddings"], dtype=np.float32)
        )
        self.prototype_mean = _normalize_vector(
            np.mean(self.prototype_embeddings, axis=0).astype(np.float32)
        )
        self._encoder: EncoderClassifier | None = None

    @property
    def encoder(self) -> EncoderClassifier:
        if self._encoder is None:
            self._encoder = EncoderClassifier.from_hparams(
                source=str(self.model_source),
                run_opts={"device": self.device},
            )
            self._encoder.eval()
        return self._encoder

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        if windows.size == 0:
            return np.zeros((0, self.prototype_embeddings.shape[1]), dtype=np.float32)

        outputs = []
        with torch.inference_mode():
            for start in range(0, len(windows), self.config.batch_size):
                batch = torch.from_numpy(windows[start:start + self.config.batch_size]).to(self.device)
                lengths = torch.ones(batch.shape[0], device=self.device)
                embedding = self.encoder.encode_batch(batch, wav_lens=lengths)
                embedding = embedding.squeeze(1).detach().cpu().numpy().astype(np.float32)
                outputs.append(embedding)
        return _normalize_rows(np.concatenate(outputs, axis=0))

    def score_audio(self, audio_path: str | Path, duration: float | None = None) -> dict:
        samples = load_audio_mono(audio_path, sample_rate=self.config.sample_rate)
        if duration is None:
            duration = len(samples) / self.config.sample_rate
        return score_amela_like_audio(samples, duration, self)


def load_audio_mono(audio_path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    src_rate, data = wavfile.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / float(scale)
    else:
        data = data.astype(np.float32)

    if src_rate != sample_rate:
        gcd = np.gcd(src_rate, sample_rate)
        data = resample_poly(data, sample_rate // gcd, src_rate // gcd).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def build_prototype_bank(
    enroll_dir: str | Path,
    out_path: str | Path = DEFAULT_PROTOTYPE_PATH,
    *,
    model_source: str = DEFAULT_MODEL_SOURCE,
    device: str | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    enroll_path = Path(enroll_dir)
    clip_paths = sorted(_iter_enrollment_wavs(enroll_path))
    if not clip_paths:
        raise SystemExit(f"No Amelia enrollment WAVs found under: {enroll_path}")

    detector = AmeliaEventDetector.__new__(AmeliaEventDetector)
    detector.prototype_path = Path(out_path)
    detector.model_source = model_source
    detector.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector.config = AmeliaEventConfig(
        sample_rate=DEFAULT_SAMPLE_RATE,
        window_sec=window_sec,
        hop_sec=hop_sec,
        score_threshold=score_threshold,
        max_gap_sec=max_gap_sec,
        batch_size=batch_size,
    )
    detector.prototype_embeddings = np.zeros((0, 192), dtype=np.float32)
    detector.prototype_mean = np.zeros(192, dtype=np.float32)
    detector._encoder = None

    embeddings = []
    clips = []
    for clip_path in clip_paths:
        samples = load_audio_mono(clip_path, sample_rate=detector.config.sample_rate)
        embedding = detector.embed_windows(samples[None, :])[0]
        embeddings.append(embedding)
        clips.append(
            {
                "path": str(clip_path),
                "duration_sec": round(len(samples) / detector.config.sample_rate, 3),
            }
        )

    prototype_embeddings = _normalize_rows(np.stack(embeddings, axis=0))
    payload = {
        "version": 1,
        "model_source": model_source,
        "config": {
            "sample_rate": detector.config.sample_rate,
            "window_sec": detector.config.window_sec,
            "hop_sec": detector.config.hop_sec,
            "score_threshold": detector.config.score_threshold,
            "max_gap_sec": detector.config.max_gap_sec,
            "batch_size": detector.config.batch_size,
        },
        "clip_count": len(clips),
        "clips": clips,
        "prototype_embeddings": prototype_embeddings.tolist(),
        "prototype_mean": _normalize_vector(np.mean(prototype_embeddings, axis=0)).tolist(),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def score_amela_like_audio(
    samples: np.ndarray,
    duration: float,
    detector: AmeliaEventDetector,
) -> dict:
    windows, records = _slice_windows(
        samples,
        sample_rate=detector.config.sample_rate,
        window_sec=detector.config.window_sec,
        hop_sec=detector.config.hop_sec,
    )
    embeddings = detector.embed_windows(windows)
    scores = _compute_similarity_scores(embeddings, detector.prototype_embeddings)

    for record, score in zip(records, scores):
        record["score"] = float(score)

    timeline = windows_to_timeline(records, duration)
    spans = merge_scored_windows(
        records,
        threshold=detector.config.score_threshold,
        max_gap_sec=detector.config.max_gap_sec,
    )

    return {
        "config": {
            "sample_rate": detector.config.sample_rate,
            "window_sec": detector.config.window_sec,
            "hop_sec": detector.config.hop_sec,
            "score_threshold": detector.config.score_threshold,
            "max_gap_sec": detector.config.max_gap_sec,
        },
        "timeline_scores": timeline.tolist(),
        "windows": records,
        "spans": spans,
    }


def windows_to_timeline(records: list[dict], duration: float) -> np.ndarray:
    length = max(1, int(duration) + 2)
    timeline = np.zeros(length, dtype=np.float32)
    counts = np.zeros(length, dtype=np.float32)

    for record in records:
        score = float(record["score"])
        start_sec = float(record["start_sec"])
        end_sec = float(record["end_sec"])
        lo = max(0, int(np.floor(start_sec)))
        hi = min(length, int(np.ceil(end_sec)) + 1)
        for idx in range(lo, hi):
            timeline[idx] = max(timeline[idx], score)
            counts[idx] += 1

    smoothed = np.convolve(timeline, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same")
    mask = counts > 0
    smoothed[~mask] = 0.0
    return smoothed


def merge_scored_windows(
    records: list[dict],
    *,
    threshold: float,
    max_gap_sec: float,
) -> list[dict]:
    active = [record for record in records if float(record["score"]) >= threshold]
    if not active:
        return []

    spans = []
    current = None
    for record in active:
        if current is None:
            current = {
                "start_sec": float(record["start_sec"]),
                "end_sec": float(record["end_sec"]),
                "peak_sec": float(record["center_sec"]),
                "peak_prob": float(record["score"]),
                "score": float(record["score"]),
            }
            continue

        if float(record["start_sec"]) - current["end_sec"] <= max_gap_sec:
            current["end_sec"] = float(record["end_sec"])
            if float(record["score"]) >= current["peak_prob"]:
                current["peak_prob"] = float(record["score"])
                current["peak_sec"] = float(record["center_sec"])
            current["score"] = max(current["score"], float(record["score"]))
        else:
            spans.append(_round_span(current))
            current = {
                "start_sec": float(record["start_sec"]),
                "end_sec": float(record["end_sec"]),
                "peak_sec": float(record["center_sec"]),
                "peak_prob": float(record["score"]),
                "score": float(record["score"]),
            }

    if current is not None:
        spans.append(_round_span(current))
    return spans


def select_top_windows(
    records: list[dict],
    *,
    threshold: float,
    max_clip_sec: float = 5.0,
) -> list[dict]:
    """
    Select raw detector windows sorted by score descending, while preventing
    overlapping duplicate clips.

    Windows below ``threshold`` are discarded. When a new high-scoring window
    overlaps an already selected clip, the two are consolidated into a single
    clip instead of producing duplicates. Consolidated clips are capped at
    ``max_clip_sec`` around the highest-score center in that cluster.
    """
    selected: list[dict] = []
    for record in sorted(records, key=lambda item: float(item["score"]), reverse=True):
        score = float(record["score"])
        if score < threshold:
            break
        start_sec = float(record["start_sec"])
        end_sec = float(record["end_sec"])
        candidate = {
            "start_sec": start_sec,
            "end_sec": min(end_sec, start_sec + max_clip_sec),
            "center_sec": float(record["center_sec"]),
            "score": score,
        }

        overlap_index = None
        for index, existing in enumerate(selected):
            if _clips_overlap_or_touch(existing, candidate):
                overlap_index = index
                break

        if overlap_index is None:
            selected.append(_round_ranked_clip(candidate))
            continue

        merged = _consolidate_ranked_clip(selected[overlap_index], candidate, max_clip_sec=max_clip_sec)
        selected[overlap_index] = _round_ranked_clip(merged)

    selected.sort(key=lambda item: float(item["score"]), reverse=True)
    return selected


def select_top_windows_for_duration(
    records: list[dict],
    *,
    target_duration_sec: float,
    max_clip_sec: float = 5.0,
) -> tuple[list[dict], float]:
    """
    Dynamically determine the minimum score needed to reach a target output
    duration, then return the selected clips sorted by time.
    """
    if target_duration_sec <= 0:
        return [], 0.0

    selected: list[dict] = []
    used_threshold = 0.0

    for record in sorted(records, key=lambda item: float(item["score"]), reverse=True):
        score = float(record["score"])
        candidate = {
            "start_sec": float(record["start_sec"]),
            "end_sec": min(float(record["end_sec"]), float(record["start_sec"]) + max_clip_sec),
            "center_sec": float(record["center_sec"]),
            "score": score,
        }

        overlap_index = None
        for index, existing in enumerate(selected):
            if _clips_overlap_or_touch(existing, candidate):
                overlap_index = index
                break

        if overlap_index is None:
            selected.append(_round_ranked_clip(candidate))
        else:
            merged = _consolidate_ranked_clip(selected[overlap_index], candidate, max_clip_sec=max_clip_sec)
            selected[overlap_index] = _round_ranked_clip(merged)

        used_threshold = score
        total_duration = sum(
            float(item["end_sec"]) - float(item["start_sec"])
            for item in selected
        )
        if total_duration >= target_duration_sec:
            break

    selected.sort(key=lambda item: float(item["start_sec"]))
    return selected, round(used_threshold, 4)


def _round_span(span: dict) -> dict:
    return {
        "start_sec": round(span["start_sec"], 3),
        "end_sec": round(span["end_sec"], 3),
        "peak_sec": round(span["peak_sec"], 3),
        "peak_prob": round(span["peak_prob"], 4),
        "score": round(span["score"], 4),
    }


def _clips_overlap_or_touch(left: dict, right: dict, epsilon: float = 1e-6) -> bool:
    return float(left["start_sec"]) <= float(right["end_sec"]) + epsilon and float(right["start_sec"]) <= float(left["end_sec"]) + epsilon


def _consolidate_ranked_clip(base: dict, incoming: dict, *, max_clip_sec: float) -> dict:
    cluster_start = min(float(base["start_sec"]), float(incoming["start_sec"]))
    cluster_end = max(float(base["end_sec"]), float(incoming["end_sec"]))

    if float(incoming["score"]) >= float(base["score"]):
        peak_center = float(incoming["center_sec"])
        peak_score = float(incoming["score"])
    else:
        peak_center = float(base["center_sec"])
        peak_score = float(base["score"])

    if cluster_end - cluster_start <= max_clip_sec:
        clip_start = cluster_start
        clip_end = cluster_end
    else:
        clip_start = peak_center - max_clip_sec / 2
        clip_end = peak_center + max_clip_sec / 2
        if clip_start > cluster_start:
            shift = min(clip_start - cluster_start, max(0.0, cluster_end - clip_end))
            clip_start -= shift
            clip_end -= shift
        if clip_end < cluster_end:
            shift = min(cluster_end - clip_end, max(0.0, clip_start - cluster_start))
            clip_start += shift
            clip_end += shift

    if clip_end - clip_start > max_clip_sec:
        clip_start = peak_center - max_clip_sec / 2
        clip_end = peak_center + max_clip_sec / 2

    return {
        "start_sec": max(0.0, clip_start),
        "end_sec": max(max(0.0, clip_start), clip_end),
        "center_sec": peak_center,
        "score": peak_score,
    }


def _round_ranked_clip(clip: dict) -> dict:
    return {
        "start_sec": round(float(clip["start_sec"]), 3),
        "end_sec": round(float(clip["end_sec"]), 3),
        "center_sec": round(float(clip["center_sec"]), 3),
        "score": round(float(clip["score"]), 4),
    }


def _iter_enrollment_wavs(enroll_dir: Path):
    prepared_dir = enroll_dir / "prepared"
    root = prepared_dir if prepared_dir.exists() else enroll_dir
    yield from root.glob("*.wav")


def _slice_windows(
    samples: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> tuple[np.ndarray, list[dict]]:
    window_len = max(1, int(round(window_sec * sample_rate)))
    hop_len = max(1, int(round(hop_sec * sample_rate)))
    total = len(samples)

    if total == 0:
        padded = np.zeros(window_len, dtype=np.float32)
        return padded[None, :], [{"start_sec": 0.0, "end_sec": window_sec, "center_sec": window_sec / 2}]

    starts = list(range(0, max(1, total - window_len + 1), hop_len))
    if not starts:
        starts = [0]
    last_start = max(0, total - window_len)
    if starts[-1] != last_start:
        starts.append(last_start)

    windows = []
    records = []
    for start in starts:
        end = min(total, start + window_len)
        window = samples[start:end]
        if len(window) < window_len:
            window = np.pad(window, (0, window_len - len(window)))
        start_sec = start / sample_rate
        end_sec = start_sec + (len(window) / sample_rate)
        records.append(
            {
                "start_sec": round(start_sec, 3),
                "end_sec": round(min(end_sec, total / sample_rate), 3),
                "center_sec": round(start_sec + window_sec / 2, 3),
            }
        )
        windows.append(window.astype(np.float32))
    return np.stack(windows, axis=0), records


def _compute_similarity_scores(
    embeddings: np.ndarray,
    prototype_embeddings: np.ndarray,
) -> np.ndarray:
    if embeddings.size == 0 or prototype_embeddings.size == 0:
        return np.zeros(len(embeddings), dtype=np.float32)

    cosine = np.clip(embeddings @ prototype_embeddings.T, -1.0, 1.0)
    top_k = min(3, cosine.shape[1])
    top_scores = np.sort(cosine, axis=1)[:, -top_k:]
    mean_top = np.mean(top_scores, axis=1)
    max_top = top_scores[:, -1]
    combined = 0.7 * max_top + 0.3 * mean_top
    probabilities = np.clip((combined + 1.0) / 2.0, 0.0, 1.0)
    return probabilities.astype(np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (matrix / norms).astype(np.float32)
