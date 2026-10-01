#!/usr/bin/env python3
"""Transcribe a folder of audio files with faster-whisper and write an audiofolder metadata.csv.

Files longer than --max-seconds are split at Whisper segment boundaries into chunks of roughly
--chunk-seconds; the chunks are written next to the source as <stem>_NNN.wav and the original
is moved to <audio-dir>_original/ so it is not loaded as a (truncated) training sample.

With --trim, every clip also has its non-speech stretches (SFX, noise, long silences) cut out
(see trim_nonspeech.py); the text comes from the same Whisper pass, and untrimmed copies of the
input files are kept in <audio-dir>_untrimmed/ (reruns trim from those copies).

The output `metadata.csv` (columns: file_name,text) is placed inside the audio directory so the
folder can be loaded with `--dataset audiofolder` in prepare_manifest.py. Review/edit it before
running the manifest step; ASR errors end up in training otherwise.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


@dataclass
class Seg:
    start: float
    end: float
    text: str


def group_segments(segs: list[Seg], target_seconds: float) -> list[list[Seg]]:
    """Greedily pack consecutive segments into groups spanning at most target_seconds."""
    groups: list[list[Seg]] = []
    cur: list[Seg] = []
    for seg in segs:
        if cur and seg.end - cur[0].start > target_seconds:
            groups.append(cur)
            cur = [seg]
        else:
            cur.append(seg)
    if cur:
        groups.append(cur)
    return groups


def transcribe_segments(model, path: Path, *, vad: bool) -> list[Seg]:
    segments, _ = model.transcribe(
        str(path),
        language="ja",
        beam_size=5,
        vad_filter=vad,
        condition_on_previous_text=False,
    )
    return [Seg(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]


def split_long_file(model, path: Path, args: argparse.Namespace, skipped: list[str]) -> list[tuple[str, str]]:
    """Split one long recording into chunks. Returns (chunk_file_name, text) rows."""
    audio, sr = sf.read(str(path), always_2d=True)
    total = audio.shape[0] / sr
    segs = transcribe_segments(model, path, vad=True)
    rows: list[tuple[str, str]] = []
    idx = 0
    for group in group_segments(segs, args.chunk_seconds):
        start = max(0.0, group[0].start - args.pad_seconds)
        end = min(total, group[-1].end + args.pad_seconds)
        dur = end - start
        text = "".join(s.text for s in group).strip()
        tag = f"{path.name}[{group[0].start:.1f}-{group[-1].end:.1f}s]"
        if dur < args.min_seconds or dur > args.max_seconds:
            skipped.append(f"{tag}\tchunk duration={dur:.2f}s")
            continue
        idx += 1
        name = f"{path.stem}_{idx:03d}.wav"
        sf.write(str(path.parent / name), audio[int(start * sr) : int(end * sr)], sr)
        rows.append((name, text))
        print(f"{name}\t{dur:.1f}s\t{text}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16", help="float16 (GPU) / int8 (CPU)")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0, help="hard limit per sample")
    parser.add_argument("--chunk-seconds", type=float, default=20.0, help="target chunk length")
    parser.add_argument("--pad-seconds", type=float, default=0.1)
    parser.add_argument("--trim", action="store_true", help="cut non-speech stretches from every clip")
    args = parser.parse_args()
    if args.chunk_seconds > args.max_seconds:
        raise SystemExit("--chunk-seconds must be <= --max-seconds")

    from faster_whisper import WhisperModel

    audio_dir = Path(args.audio_dir)
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"no audio files in {audio_dir}")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    original_dir = audio_dir.parent / f"{audio_dir.name}_original"
    untrimmed_dir = audio_dir.parent / f"{audio_dir.name}_untrimmed"
    if args.trim:
        from trim_nonspeech import TrimConfig, trim_clip

        trim_cfg = TrimConfig()

    def trim(src: Path, dst: Path) -> tuple[str, float] | None:
        text, dur_in, dur_out = trim_clip(model, src, dst, trim_cfg)
        if dur_out == 0:
            return None
        print(f"  trim {dst.name}: {dur_in:.1f}s -> {dur_out:.1f}s", flush=True)
        return text, dur_out

    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for path in files:
        dur = sf.info(str(path)).duration
        if dur > args.max_seconds:
            chunks = split_long_file(model, path, args, skipped)
            original_dir.mkdir(exist_ok=True)
            shutil.move(str(path), str(original_dir / path.name))
            if not args.trim:
                rows.extend(chunks)
                continue
            for name, _ in chunks:
                chunk = audio_dir / name
                res = trim(chunk, chunk)
                if res is None or res[1] < args.min_seconds:
                    skipped.append(f"{name}\tno speech after trim")
                    chunk.unlink()
                    continue
                rows.append((name, res[0]))
            continue
        if dur < args.min_seconds:
            skipped.append(f"{path.name}\tduration={dur:.2f}s")
            continue
        if args.trim:
            untrimmed_dir.mkdir(exist_ok=True)
            backup = untrimmed_dir / path.name
            if not backup.exists():  # on reruns, re-trim from the untouched copy
                shutil.copy2(str(path), str(backup))
            res = trim(backup, path)
            if res is None or res[1] < args.min_seconds:
                skipped.append(f"{path.name}\tno speech after trim")
                continue
            text = res[0]
        else:
            segs = transcribe_segments(model, path, vad=False)
            text = "".join(s.text for s in segs).strip()
        if not text:
            skipped.append(f"{path.name}\tempty transcript")
            continue
        rows.append((path.name, text))
        print(f"{path.name}\t{dur:.1f}s\t{text}", flush=True)

    out = audio_dir / "metadata.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerows(rows)

    # Keep this outside the audio dir so audiofolder does not pick it up.
    skipped_path = audio_dir.parent / f"{audio_dir.name}_skipped.tsv"
    skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}; skipped {len(skipped)} (see {skipped_path})")


if __name__ == "__main__":
    main()
