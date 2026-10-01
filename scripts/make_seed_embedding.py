#!/usr/bin/env python3
"""Write a fixed random Speaker Inversion embedding to use as a shared training seed.

Speakers trained from the same seed (scripts/train_voice.sh NAME --init-embed SEED) start from
identical tokens instead of each drawing its own random init, so their tokens have a chance to
stay aligned and the resulting embeddings can be interpolated. The seed carries no speaker
information: it is drawn from the same N(0, init_std) as the default random init.

Token count, dim and std default to the Speaker Inversion config (Large unless --config says
otherwise).

  uv run --no-sync python scripts/make_seed_embedding.py --seed 0
  -> outputs/speaker_inversion/seeds/seed0_large.speaker.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irodori_tts.speaker_inversion import save_speaker_inversion_safetensors  # noqa: E402

DEFAULT_CONFIG = "configs/train_v4_large_speaker_inversion.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed (default: 0)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    parser.add_argument("--tokens", type=int, help="token count (default: from config)")
    parser.add_argument("--std", type=float, help="init stddev (default: from config)")
    parser.add_argument(
        "--output",
        help="output path (default: outputs/speaker_inversion/seeds/seed<SEED>_<MODEL>"
        ".speaker.safetensors)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    speaker_dim = int(cfg["model"]["speaker_dim"])
    tokens = args.tokens if args.tokens is not None else int(cfg["train"]["speaker_inversion_tokens"])
    std = args.std if args.std is not None else float(cfg["train"]["speaker_inversion_init_std"])
    model = "small" if "small" in Path(args.config).name else "large"
    output = args.output or (
        f"outputs/speaker_inversion/seeds/seed{args.seed}_{model}.speaker.safetensors"
    )

    generator = torch.Generator().manual_seed(args.seed)
    embedding = torch.randn(tokens, speaker_dim, generator=generator) * std
    save_speaker_inversion_safetensors(output, {"speaker_embedding": embedding})
    print(f"{output}: {tokens} tokens x {speaker_dim} dim, std={std}, seed={args.seed}")


if __name__ == "__main__":
    main()
