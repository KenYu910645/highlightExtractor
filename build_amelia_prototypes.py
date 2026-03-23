#!/usr/bin/env python3
"""
Build Amelia prototype embeddings from prepared enrollment WAV clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from amelia_event import (
    DEFAULT_MODEL_SOURCE,
    DEFAULT_PROTOTYPE_PATH,
    build_prototype_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Amelia event prototype embeddings.")
    parser.add_argument(
        "--enroll-dir",
        default="data/enroll/amelia",
        help="Directory containing Amelia enrollment clips (default: data/enroll/amelia)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_PROTOTYPE_PATH,
        help=f"Output JSON artifact path (default: {DEFAULT_PROTOTYPE_PATH})",
    )
    parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_SOURCE,
        help=f"SpeechBrain model source/checkpoint directory (default: {DEFAULT_MODEL_SOURCE})",
    )
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    payload = build_prototype_bank(
        enroll_dir=args.enroll_dir,
        out_path=args.out,
        model_source=args.model_source,
        device=args.device,
    )

    print("Amelia prototype bank ready")
    print(f"  Enroll dir : {Path(args.enroll_dir).resolve()}")
    print(f"  Output     : {Path(args.out).resolve()}")
    print(f"  Clips      : {payload['clip_count']}")
    print(f"  Model      : {payload['model_source']}")


if __name__ == "__main__":
    main()
