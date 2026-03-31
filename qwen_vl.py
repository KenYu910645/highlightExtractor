#!/usr/bin/env python3
"""
qwen_vl.py - Shared local Qwen2.5-VL helpers.
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def resolve_image_path(image_path: str | Path) -> Path:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    return path


def path_to_file_uri(path: Path) -> str:
    return f"file:///{quote(path.as_posix(), safe='/:')}"


def build_messages(image_uri: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def choose_torch_dtype(torch_module: Any) -> Any:
    if torch_module.cuda.is_available():
        return torch_module.bfloat16
    return torch_module.float32


def load_model_and_processor(model_name: str, max_pixels: int | None):
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for Qwen vision tools. Install a recent "
            "`transformers` build plus `accelerate` and `Pillow` before running "
            "this workflow."
        ) from exc

    processor_kwargs: dict[str, Any] = {}
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=choose_torch_dtype(torch),
        device_map="auto",
    )
    return model, processor


@contextlib.contextmanager
def suppress_runtime_noise():
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*loaded as a fast processor by default.*")
        warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*")
        warnings.filterwarnings("ignore", message=r".*cache-system uses symlinks by default.*")
        with contextlib.redirect_stderr(stderr_buffer):
            try:
                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()
            except Exception:
                pass
            try:
                from transformers.utils import logging as transformers_logging

                transformers_logging.set_verbosity_error()
            except Exception:
                pass
            yield


def load_local_image(image_path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for Qwen vision tools. Install `Pillow` before running this workflow."
        ) from exc
    return Image.open(image_path).convert("RGB")


def extract_json_candidate(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index : index + end]
        except json.JSONDecodeError:
            continue
    return None


def generate_answer(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    image_path: Path,
    max_new_tokens: int,
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[load_local_image(image_path)],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0]
