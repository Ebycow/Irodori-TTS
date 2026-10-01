#!/usr/bin/env bash
# One-shot Speaker Inversion workflow:
#   data/<NAME>/*.wav  ->  Whisper transcripts  ->  latents + manifest  ->  speaker embedding
#
# Usage:
#   scripts/train_voice.sh NAME [--review] [--trim] [--rebuild] [--force] [--small]
#                               [--scale-steps] [--grad-ckpt] [--init-embed PATH]
#
#   --review   stop after transcription so metadata.csv can be corrected; rerun to continue
#   --trim     cut non-speech stretches (SFX/noise/silence) from every clip while transcribing;
#              untrimmed copies go to data/NAME_untrimmed/
#   --rebuild  redo latents + manifest only (after editing audio in data/NAME by hand);
#              keeps metadata.csv
#   --force    redo every step even if its output already exists (overwrites metadata.csv)
#   --small    use the v4-Small config/checkpoint instead of v4-Large
#   --scale-steps  divide the config's max_steps by NGPU (same samples seen, shorter run);
#                  without it, extra GPUs only grow the effective batch size
#   --grad-ckpt    keep gradient checkpointing on (off by default: faster, needs more VRAM)
#   --init-embed PATH  start from a shared .speaker.safetensors instead of random tokens, so
#                  speakers trained from the same base stay token-aligned and can be blended;
#                  the token count follows the file, and output goes to
#                  outputs/speaker_inversion/NAME_MODEL_from-BASE unless OUT_DIR is set
#
# Check transcripts against the audio: scripts/verify_transcripts.py data/NAME
# Environment overrides: CKPT, CONFIG, OUT_DIR, WHISPER_ARGS (e.g. "--device cpu --compute-type int8"),
#   NGPU (GPUs for manifest prep + DDP training; default: SLURM_GPUS_ON_NODE under srun, else the
#   number of entries in CUDA_VISIBLE_DEVICES, else 1).
# Needs HF login (google/t5gemma-2-1b-1b is gated) and FFmpeg shared libs on LD_LIBRARY_PATH.

set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-}"
[[ -n "$NAME" && "$NAME" != --* ]] || { sed -n 2,28p "$0"; exit 1; }
shift

REVIEW=0; FORCE=0; REBUILD=0; TRIM_ARGS=; MODEL=large; SCALE_STEPS=0; GRAD_CKPT=--no-gradient-checkpointing
INIT_EMBED=
if [[ -z "${NGPU:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NGPU="$SLURM_GPUS_ON_NODE"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"
    NGPU="${#_gpus[@]}"
  else
    NGPU=1
  fi
fi
while [[ $# -gt 0 ]]; do
  arg="$1"; shift
  case "$arg" in
    --review) REVIEW=1 ;;
    --force) FORCE=1 ;;
    --rebuild) REBUILD=1 ;;
    --trim) TRIM_ARGS=--trim ;;
    --small) MODEL=small ;;
    --scale-steps) SCALE_STEPS=1 ;;
    --grad-ckpt) GRAD_CKPT=--gradient-checkpointing ;;
    --init-embed)
      [[ $# -gt 0 ]] || { echo "--init-embed needs a path" >&2; exit 1; }
      INIT_EMBED="$1"; shift ;;
    --init-embed=*) INIT_EMBED="${arg#*=}" ;;
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
if [[ -n "$INIT_EMBED" ]]; then
  [[ -f "$INIT_EMBED" ]] || { echo "init embedding not found: $INIT_EMBED" >&2; exit 1; }
  [[ "$INIT_EMBED" == *.speaker.safetensors ]] || {
    echo "init embedding must be a .speaker.safetensors file: $INIT_EMBED" >&2; exit 1; }
  # checkpoint_*.speaker.safetensors says nothing about the base, so name it after its run dir.
  INIT_STEM="$(basename "$INIT_EMBED" .speaker.safetensors)"
  [[ "$INIT_STEM" == checkpoint_* ]] && INIT_STEM="$(basename "$(dirname "$INIT_EMBED")")"
  OUT_DIR="${OUT_DIR:-outputs/speaker_inversion/${NAME}_${MODEL}_from-${INIT_STEM}}"
else
  OUT_DIR="${OUT_DIR:-outputs/speaker_inversion/${NAME}_${MODEL}}"
fi

[[ -d "$AUDIO_DIR" ]] || { echo "missing directory: $AUDIO_DIR" >&2; exit 1; }
found=0
for ext in wav flac mp3 ogg; do
  compgen -G "$AUDIO_DIR/*.$ext" > /dev/null && found=1
done
[[ $found -eq 1 ]] || { echo "no .wav/.flac/.mp3/.ogg files in $AUDIO_DIR" >&2; exit 1; }
[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT (set CKPT=...)" >&2; exit 1; }
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo "NGPU must be a positive integer: $NGPU" >&2; exit 1; }

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
    --device cuda \
    --num-gpus "$NGPU" \
    --merge-output
else
  echo "manifest exists, skipping (use --rebuild to redo)"
fi

TRAIN_ARGS=(--config "$CONFIG" --manifest "$MANIFEST" --init-checkpoint "$CKPT" --output-dir "$OUT_DIR" "$GRAD_CKPT")
if [[ $SCALE_STEPS -eq 1 ]]; then
  MAX_STEPS=$(awk '/^[[:space:]]*max_steps:/ {print $2; exit}' "$CONFIG")
  [[ -n "$MAX_STEPS" ]] || { echo "max_steps not found in $CONFIG" >&2; exit 1; }
  TRAIN_ARGS+=(--max-steps $(( (MAX_STEPS + NGPU - 1) / NGPU )))
fi
if [[ -n "$INIT_EMBED" ]]; then
  INIT_TOKENS=$(uv run --no-sync python -c \
    'import sys; from safetensors import safe_open; print(safe_open(sys.argv[1], "pt").get_slice("speaker_embedding").get_shape()[-2])' \
    "$INIT_EMBED")
  echo "init embedding: $INIT_EMBED ($INIT_TOKENS tokens)"
  TRAIN_ARGS+=(--speaker-inversion-init-embedding "$INIT_EMBED" --speaker-inversion-tokens "$INIT_TOKENS")
fi

echo "== [3/3] train ($MODEL, ${NGPU} GPU): $OUT_DIR"
if [[ "$NGPU" -gt 1 ]]; then
  uv run --no-sync torchrun --standalone --nproc_per_node "$NGPU" train.py "${TRAIN_ARGS[@]}"
else
  uv run --no-sync python train.py "${TRAIN_ARGS[@]}"
fi

echo "Done: $OUT_DIR/checkpoint_final.speaker.safetensors"
