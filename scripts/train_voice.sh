#!/usr/bin/env bash
# One-shot Speaker Inversion workflow:
#   data/<NAME>/*.wav  ->  Whisper transcripts  ->  latents + manifest  ->  speaker embedding
#
# Usage:
#   scripts/train_voice.sh NAME [--review] [--trim] [--rebuild] [--force] [--small]
#
#   --review   stop after transcription so metadata.csv can be corrected; rerun to continue
#   --trim     cut non-speech stretches (SFX/noise/silence) from every clip while transcribing;
#              untrimmed copies go to data/NAME_untrimmed/
#   --rebuild  redo latents + manifest only (after editing audio in data/NAME by hand);
#              keeps metadata.csv
#   --force    redo every step even if its output already exists (overwrites metadata.csv)
#   --small    use the v4-Small config/checkpoint instead of v4-Large
#
# Check transcripts against the audio: scripts/verify_transcripts.py data/NAME
# Environment overrides: CKPT, CONFIG, WHISPER_ARGS (e.g. "--device cpu --compute-type int8").
# Needs HF login (google/t5gemma-2-1b-1b is gated) and FFmpeg shared libs on LD_LIBRARY_PATH.

set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-}"
[[ -n "$NAME" && "$NAME" != --* ]] || { sed -n 2,20p "$0"; exit 1; }
shift

REVIEW=0; FORCE=0; REBUILD=0; TRIM_ARGS=; MODEL=large
for arg in "$@"; do
  case "$arg" in
    --review) REVIEW=1 ;;
    --force) FORCE=1 ;;
    --rebuild) REBUILD=1 ;;
    --trim) TRIM_ARGS=--trim ;;
    --small) MODEL=small ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$MODEL" == large ]]; then
  CONFIG="${CONFIG:-configs/train_v4_large_speaker_inversion.yaml}"
  CKPT="${CKPT:-models/Irodori-TTS-v4-Large/model.safetensors}"
else
  CONFIG="${CONFIG:-configs/train_v4_small_speaker_inversion.yaml}"
  CKPT="${CKPT:-models/Irodori-TTS-v4.1-Small/model.safetensors}"
fi

AUDIO_DIR="data/$NAME"
MANIFEST="data/${NAME}_manifest.jsonl"
LATENT_DIR="data/${NAME}_latents"
OUT_DIR="outputs/speaker_inversion/${NAME}_${MODEL}"

[[ -d "$AUDIO_DIR" ]] || { echo "missing directory: $AUDIO_DIR" >&2; exit 1; }
found=0
for ext in wav flac mp3 ogg; do
  compgen -G "$AUDIO_DIR/*.$ext" > /dev/null && found=1
done
[[ $found -eq 1 ]] || { echo "no .wav/.flac/.mp3/.ogg files in $AUDIO_DIR" >&2; exit 1; }
[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT (set CKPT=...)" >&2; exit 1; }

echo "== [1/3] transcribe: $AUDIO_DIR"
if [[ $FORCE -eq 1 || ! -f "$AUDIO_DIR/metadata.csv" ]]; then
  # shellcheck disable=SC2086
  uv run --no-project --with faster-whisper --with soundfile \
    python scripts/transcribe_whisper.py --audio-dir "$AUDIO_DIR" $TRIM_ARGS ${WHISPER_ARGS:-}
  if [[ $REVIEW -eq 1 ]]; then
    echo "Review/fix $AUDIO_DIR/metadata.csv, then rerun: scripts/train_voice.sh $NAME"
    exit 0
  fi
else
  echo "metadata.csv exists, skipping (use --force to redo)"
  [[ -z "$TRIM_ARGS" ]] || echo "warning: --trim only applies when transcribing; use --force, or run scripts/trim_nonspeech.py" >&2
fi

echo "== [2/3] prepare manifest: $MANIFEST"
if [[ $FORCE -eq 1 || $REBUILD -eq 1 || ! -s "$MANIFEST" ]]; then
  # Stale latents from an earlier run would otherwise linger next to the new ones.
  rm -rf "$LATENT_DIR" "$MANIFEST" "$MANIFEST".*irodori_index.pt
  uv run --no-sync python prepare_manifest.py \
    --dataset audiofolder \
    --data-files "$AUDIO_DIR/**" \
    --split train \
    --audio-column audio \
    --text-column text \
    --output-manifest "$MANIFEST" \
    --latent-dir "$LATENT_DIR" \
    --device cuda
else
  echo "manifest exists, skipping (use --rebuild to redo)"
fi

echo "== [3/3] train ($MODEL): $OUT_DIR"
uv run --no-sync python train.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --init-checkpoint "$CKPT" \
  --output-dir "$OUT_DIR"

echo "Done: $OUT_DIR/checkpoint_final.speaker.safetensors"
