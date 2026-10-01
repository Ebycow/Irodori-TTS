"""Cut non-speech stretches (SFX, noise, long silences) from short clips without source separation.

Speech = union of Silero VAD regions and Whisper word timestamps, so quiet interjections that
VAD misses are kept. Each speech region gets --pad seconds of context; gaps shorter than
--merge seconds stay untouched; longer ones are dropped and the joins crossfaded. Kept audio is
copied sample-for-sample, so quality is unchanged.

Also used by transcribe_whisper.py --trim. Standalone use (for an already transcribed folder):

  uv run --no-project --with faster-whisper --with soundfile \
    python scripts/trim_nonspeech.py data/NAME data/NAME_trimmed
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class TrimConfig:
    pad: float = 0.2
    merge: float = 0.6
    crossfade: float = 0.02
    vad_threshold: float = 0.5


def to16k(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def merge(regions: list[tuple[float, float]], pad: float, gap: float, dur: float) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in sorted(regions):
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if out and s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def splice(x: np.ndarray, sr: int, keep: list[list[float]], xf_sec: float) -> np.ndarray:
    xf = int(xf_sec * sr)
    out = x[int(keep[0][0] * sr) : int(keep[0][1] * sr)].copy()
    for s, e in keep[1:]:
        seg = x[int(s * sr) : int(e * sr)]
        ramp = np.linspace(0, 1, xf, dtype=np.float32)
        if x.ndim == 2:
            ramp = ramp[:, None]
        out[-xf:] = out[-xf:] * (1 - ramp) + seg[:xf] * ramp
        out = np.concatenate([out, seg[xf:]])
    return out


def trim_clip(model, src: Path, dst: Path, cfg: TrimConfig) -> tuple[str, float, float]:
    """Transcribe src, write the speech-only version to dst. Returns (text, dur_in, dur_out).

    dst may equal src. Nothing is written when no speech is found (dur_out == 0).
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    x, sr = sf.read(str(src), dtype="float32")
    mono = x.mean(1) if x.ndim == 2 else x
    dur = len(mono) / sr
    vad_opts = VadOptions(threshold=cfg.vad_threshold, min_silence_duration_ms=150, speech_pad_ms=0)
    regions = [(t["start"] / 16000, t["end"] / 16000) for t in get_speech_timestamps(to16k(mono, sr), vad_opts)]
    segs, _ = model.transcribe(
        str(src), language="ja", beam_size=5, word_timestamps=True, condition_on_previous_text=False
    )
    segs = list(segs)
    text = "".join(s.text.strip() for s in segs).strip()
    regions += [(w.start, w.end) for s in segs for w in (s.words or [])]
    keep = merge(regions, cfg.pad, cfg.merge, dur)
    if not keep:
        return text, dur, 0.0
    out = splice(x, sr, keep, cfg.crossfade)
    sf.write(str(dst), out, sr, subtype="PCM_16")
    return text, dur, len(out) / sr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--pad", type=float, default=TrimConfig.pad)
    p.add_argument("--merge", type=float, default=TrimConfig.merge)
    p.add_argument("--crossfade", type=float, default=TrimConfig.crossfade)
    p.add_argument("--vad-threshold", type=float, default=TrimConfig.vad_threshold)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    args = p.parse_args()

    from faster_whisper import WhisperModel

    cfg = TrimConfig(args.pad, args.merge, args.crossfade, args.vad_threshold)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    names = [r["file_name"] for r in csv.DictReader(open(src / "metadata.csv", encoding="utf-8"))]
    total_in = total_out = 0.0
    for name in names:
        _, dur, out_dur = trim_clip(model, src / name, dst / name, cfg)
        if out_dur == 0:
            print(f"{name}: no speech found, copied as-is", flush=True)
            shutil.copy(src / name, dst / name)
            out_dur = dur
        total_in += dur
        total_out += out_dur
        print(f"{name}: {dur:.2f}s -> {out_dur:.2f}s", flush=True)
    shutil.copy(src / "metadata.csv", dst / "metadata.csv")
    print(f"total {total_in:.1f}s -> {total_out:.1f}s")


if __name__ == "__main__":
    main()
