"""Embedding model wrappers for offline speaker labeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32, copy=True)
    return (vec / norm).astype(np.float32, copy=False)


@dataclass
class SpeakerEmbeddingModel:
    """Lazy wrapper for a local pretrained speaker embedding model."""

    model_name: str = "speechbrain-ecapa"
    _classifier: object | None = None
    _torch: object | None = None

    def _load(self):
        if self._classifier is not None:
            return self._classifier, self._torch

        if self.model_name not in {"speechbrain-ecapa", "auto"}:
            raise ValueError(f"Unsupported speaker model: {self.model_name}")

        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:
            raise RuntimeError(
                "Speaker labeling requires SpeechBrain and PyTorch. "
                "Install them with a Python interpreter, for example: "
                "pip install torch torchaudio speechbrain"
            ) from exc

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
        self._classifier = classifier
        self._torch = torch
        return classifier, torch

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if len(samples) == 0:
            raise ValueError("Cannot embed an empty audio segment")

        classifier, torch = self._load()
        wav = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        embedding = classifier.encode_batch(wav).detach().cpu().numpy().reshape(-1)
        return normalize_vector(embedding)
