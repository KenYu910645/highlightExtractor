#!/usr/bin/env python3
"""
analyze_image.py - Manual single-image analysis with Qwen2.5-VL-7B.

This is a standalone side tool for debugging and exploration. It is not part
of the automatic highlight extraction pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from qwen_vl import (
    DEFAULT_MODEL,
    build_messages as build_base_messages,
    extract_json_candidate,
    generate_answer,
    load_model_and_processor,
    path_to_file_uri,
    resolve_image_path,
    suppress_runtime_noise,
)


DEFAULT_PROMPT = "Describe this image in detail."
DEFAULT_MAX_NEW_TOKENS = 256
JSON_FALLBACK_MESSAGE = (
    "WARNING: Model output was not valid JSON. Returning raw text under "
    '"raw_output" with "validated_json": false.'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one local image with Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("image", help="Path to a local image file")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f'Prompt text (default: "{DEFAULT_PROMPT}")',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Request best-effort JSON output",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the output text or JSON",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional max pixel budget passed to AutoProcessor for VRAM control",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL})",
    )
    return parser

def build_messages(image_uri: str, prompt: str, json_mode: bool) -> list[dict[str, Any]]:
    text_prompt = prompt.strip() or DEFAULT_PROMPT
    if json_mode:
        text_prompt = (
            f"{text_prompt}\n\n"
            "Return JSON only. Use a single JSON object with this shape: "
            '{"description": string, "notable_details": [string], '
            '"confidence": number}.'
        )

    return build_base_messages(image_uri, text_prompt)


def format_response(raw_text: str, json_mode: bool) -> tuple[str, bool]:
    cleaned = raw_text.strip()
    if not json_mode:
        return cleaned, False

    candidate = extract_json_candidate(cleaned)
    if candidate is None:
        fallback = {
            "validated_json": False,
            "warning": JSON_FALLBACK_MESSAGE,
            "raw_output": cleaned,
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2), False

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        fallback = {
            "validated_json": False,
            "warning": JSON_FALLBACK_MESSAGE,
            "raw_output": cleaned,
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2), False

    return json.dumps(parsed, ensure_ascii=False, indent=2), True

def write_output(output_text: str, output_path: str | None) -> Path | None:
    if output_path is None:
        return None
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output_text + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than 0")
    if args.max_pixels is not None and args.max_pixels <= 0:
        parser.error("--max-pixels must be greater than 0")

    try:
        image_path = resolve_image_path(args.image)
        image_uri = path_to_file_uri(image_path)
        messages = build_messages(image_uri, args.prompt, args.json)
        with suppress_runtime_noise():
            model, processor = load_model_and_processor(args.model, args.max_pixels)
            raw_answer = generate_answer(
                model=model,
                processor=processor,
                messages=messages,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
            )
        output_text, _ = format_response(raw_answer, args.json)
        write_output(output_text, args.out)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
