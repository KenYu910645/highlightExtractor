"""Speaker labeling pipeline for subtitle segments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import (
    decode_audio_to_array,
    estimate_voiced_seconds,
    is_supported_audio_file,
    is_supported_source_file,
    slice_audio,
)
from .embeddings import SpeakerEmbeddingModel, normalize_vector


class SpeakerLabelingError(RuntimeError):
    """Raised when speaker labeling cannot proceed."""


@dataclass
class SpeakerLabelingConfig:
    enroll_dir: Path
    speaker_model: str = "speechbrain-ecapa"
    speaker_threshold: float = 0.58
    min_voice_sec: float = 0.35
    smoothing_margin: float = 0.08
    cache_filename: str = ".speaker_prototypes.json"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(normalize_vector(vec_a), normalize_vector(vec_b)))


def smooth_labels(segments: list[dict], margin: float = 0.08) -> list[dict]:
    """
    Prevent one low-confidence subtitle from flipping between the same neighbors.
    """
    if len(segments) < 3:
        return segments

    smoothed = [dict(seg) for seg in segments]
    for index in range(1, len(smoothed) - 1):
        prev_seg = smoothed[index - 1]
        curr_seg = smoothed[index]
        next_seg = smoothed[index + 1]
        if prev_seg.get("speaker") != next_seg.get("speaker"):
            continue
        if prev_seg.get("speaker") in {None, "unknown"}:
            continue
        if curr_seg.get("speaker") == prev_seg.get("speaker"):
            continue

        current_conf = float(curr_seg.get("speaker_confidence", 0.0) or 0.0)
        neighbor_conf = min(
            float(prev_seg.get("speaker_confidence", 0.0) or 0.0),
            float(next_seg.get("speaker_confidence", 0.0) or 0.0),
        )
        if current_conf + margin < neighbor_conf:
            curr_seg["speaker"] = prev_seg["speaker"]
            curr_seg["speaker_confidence"] = current_conf
            curr_seg["speaker_smoothed"] = True

    return smoothed


class SpeakerLabelingPipeline:
    """Classify subtitle segments against reusable speaker prototypes."""

    def __init__(self, config: SpeakerLabelingConfig):
        self.config = config
        self.model = SpeakerEmbeddingModel(config.speaker_model)

    def _speaker_dirs(self) -> dict[str, Path]:
        return {
            "Ken": self.config.enroll_dir / "ken",
            "Amelia": self.config.enroll_dir / "amelia",
        }

    def _collect_files(self, speaker_dir: Path) -> list[Path]:
        if not speaker_dir.exists():
            return []
        prepared_dir = speaker_dir / "prepared"
        if prepared_dir.exists():
            prepared_files = [path for path in prepared_dir.iterdir() if is_supported_audio_file(path)]
            if prepared_files:
                return sorted(prepared_files)

        files = [path for path in speaker_dir.iterdir() if is_supported_source_file(path)]
        return sorted(files)

    def _cache_path(self) -> Path:
        return self.config.enroll_dir / self.config.cache_filename

    def _cache_payload(self, files_by_speaker: dict[str, list[Path]], prototypes: dict[str, np.ndarray]) -> dict:
        return {
            "speaker_model": self.config.speaker_model,
            "speaker_threshold": self.config.speaker_threshold,
            "speakers": {
                speaker: {
                    "files": [
                        {
                            "path": str(path.resolve()),
                            "mtime_ns": path.stat().st_mtime_ns,
                            "size": path.stat().st_size,
                        }
                        for path in files
                    ],
                    "prototype": prototypes[speaker].tolist(),
                }
                for speaker, files in files_by_speaker.items()
            },
        }

    def _load_cached_prototypes(self, files_by_speaker: dict[str, list[Path]]) -> dict[str, np.ndarray] | None:
        cache_path = self._cache_path()
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if payload.get("speaker_model") != self.config.speaker_model:
            return None

        speakers_payload = payload.get("speakers", {})
        expected_speakers = set(files_by_speaker)
        if set(speakers_payload) != expected_speakers:
            return None

        for speaker, files in files_by_speaker.items():
            cached_files = speakers_payload.get(speaker, {}).get("files", [])
            current_files = [
                {
                    "path": str(path.resolve()),
                    "mtime_ns": path.stat().st_mtime_ns,
                    "size": path.stat().st_size,
                }
                for path in files
            ]
            if cached_files != current_files:
                return None

        prototypes = {}
        for speaker, info in speakers_payload.items():
            prototype = np.asarray(info.get("prototype", []), dtype=np.float32)
            if prototype.size == 0:
                return None
            prototypes[speaker] = normalize_vector(prototype)
        return prototypes

    def _save_cached_prototypes(self, files_by_speaker: dict[str, list[Path]], prototypes: dict[str, np.ndarray]) -> None:
        cache_path = self._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._cache_payload(files_by_speaker, prototypes)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_prototypes(self) -> dict[str, np.ndarray]:
        speaker_dirs = self._speaker_dirs()
        files_by_speaker = {speaker: self._collect_files(path) for speaker, path in speaker_dirs.items()}

        missing = [speaker for speaker, files in files_by_speaker.items() if not files]
        if missing:
            raise SpeakerLabelingError(
                "Missing enrollment audio for: "
                + ", ".join(missing)
                + f". Add clean clips under {self.config.enroll_dir}."
            )

        cached = self._load_cached_prototypes(files_by_speaker)
        if cached is not None:
            return cached

        prototypes: dict[str, np.ndarray] = {}
        for speaker, files in files_by_speaker.items():
            embeddings = []
            for path in files:
                samples, sample_rate = decode_audio_to_array(str(path))
                if estimate_voiced_seconds(samples, sample_rate) < self.config.min_voice_sec:
                    continue
                embeddings.append(self.model.embed(samples, sample_rate))

            if not embeddings:
                raise SpeakerLabelingError(
                    f"No usable voiced enrollment clips found for {speaker} in {speaker_dirs[speaker]}"
                )
            prototype = normalize_vector(np.mean(np.vstack(embeddings), axis=0))
            prototypes[speaker] = prototype

        self._save_cached_prototypes(files_by_speaker, prototypes)
        return prototypes

    def classify_segments(self, audio_path: str, segments: list[dict]) -> list[dict]:
        prototypes = self.build_prototypes()
        samples, sample_rate = decode_audio_to_array(audio_path)

        labeled_segments = []
        for seg in segments:
            labeled = dict(seg)
            chunk = slice_audio(samples, sample_rate, float(seg["start"]), float(seg["end"]))
            voice_sec = estimate_voiced_seconds(chunk, sample_rate)

            labeled["speaker"] = "unknown"
            labeled["speaker_confidence"] = 0.0
            labeled["speaker_scores"] = {}
            labeled["speaker_voice_sec"] = round(float(voice_sec), 3)

            if voice_sec < self.config.min_voice_sec or len(chunk) == 0:
                labeled_segments.append(labeled)
                continue

            embedding = self.model.embed(chunk, sample_rate)
            scores = {speaker: cosine_similarity(embedding, prototype) for speaker, prototype in prototypes.items()}
            best_speaker, best_score = max(scores.items(), key=lambda item: item[1])
            sorted_scores = sorted(scores.values(), reverse=True)
            runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            confidence = best_score - runner_up

            labeled["speaker_scores"] = {speaker: round(score, 4) for speaker, score in scores.items()}
            labeled["speaker_confidence"] = round(float(confidence), 4)
            if best_score >= self.config.speaker_threshold and confidence > 0:
                labeled["speaker"] = best_speaker

            labeled_segments.append(labeled)

        return smooth_labels(labeled_segments, margin=self.config.smoothing_margin)
