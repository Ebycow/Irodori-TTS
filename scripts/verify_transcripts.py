#!/usr/bin/env python3
"""Re-transcribe every clip in metadata.csv and compare with the stored text.

Prints `similarity<TAB>file<TAB>metadata text<TAB>whisper text`, worst first. Low scores point at
clips whose metadata is wrong or whose speech was cut by trimming (kana/kanji spelling
differences also lower the score, so read the texts rather than trusting the number).

  uv run --no-project --with faster-whisper --with soundfile \
    python scripts/verify_transcripts.py data/NAME > data/NAME_verify.tsv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
from pathlib import Path

PUNCT = re.compile(r"[、。,.?!？！\s「」…ー〜]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio_dir")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16", help="float16 (GPU) / int8 (CPU)")
    args = p.parse_args()

    from faster_whisper import WhisperModel

    audio_dir = Path(args.audio_dir)
    meta = {r["file_name"]: r["text"] for r in csv.DictReader(open(audio_dir / "metadata.csv", encoding="utf-8"))}
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    results = []
    for name, ref in meta.items():
        segs, _ = model.transcribe(str(audio_dir / name), language="ja", beam_size=5, condition_on_previous_text=False)
        hyp = "".join(s.text.strip() for s in segs)
        score = difflib.SequenceMatcher(None, PUNCT.sub("", ref), PUNCT.sub("", hyp)).ratio()
        results.append((score, name, ref, hyp))
    for score, name, ref, hyp in sorted(results):
        print(f"{score:.2f}\t{name}\t{ref}\t{hyp}")


if __name__ == "__main__":
    main()
