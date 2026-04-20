#!/usr/bin/env python3
"""FastAPI server exposing Irodori-TTS inference as an HTTP API for SillyTavern integration."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
)

app = FastAPI(title="Irodori-TTS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime: InferenceRuntime | None = None
_ref_dir: Path = Path("references")
_model_type: str = "base"  # "base" or "voicedesign"

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# --------------------------------------------------------------------------- #
#  Request / Response models                                                   #
# --------------------------------------------------------------------------- #

class TtsRequest(BaseModel):
    text: str
    voice_id: str = "no_ref"
    caption: str | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    seed: int | None = None
    seconds: float = 30.0
    trim_tail: bool = True


class VoiceObject(BaseModel):
    name: str
    voice_id: str
    preview_url: str | None = None
    lang: str = "ja"


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _audio_to_wav_bytes(audio, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    try:
        torchaudio.save(buf, audio, sample_rate, format="wav")
    except RuntimeError:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio.squeeze(0).numpy(), sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def _resolve_ref_wav(voice_id: str) -> str | None:
    """Strip ref_ prefix and return absolute path, or None for no_ref."""
    if voice_id == "no_ref" or not voice_id:
        return None
    filename = voice_id.removeprefix("ref_")
    ref_path = _ref_dir / filename
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
    return str(ref_path)


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ready",
        "model_type": _model_type,
        "ref_dir": str(_ref_dir),
    }


@app.get("/voices", response_model=list[VoiceObject])
def list_voices():
    voices: list[VoiceObject] = []

    voices.append(VoiceObject(name="[No Reference]", voice_id="no_ref", lang="ja"))

    print(f"[voices] scanning ref_dir: {_ref_dir} (exists={_ref_dir.exists()})", flush=True)
    if _ref_dir.exists():
        found = []
        for p in sorted(_ref_dir.iterdir()):
            print(f"[voices]   found: {p.name!r}  is_file={p.is_file()}  suffix={p.suffix.lower()!r}", flush=True)
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                found.append(p)
                voices.append(VoiceObject(
                    name=f"[Ref] {p.name}",
                    voice_id=f"ref_{p.name}",
                    lang="ja",
                ))
        print(f"[voices] {len(found)} reference file(s) added.", flush=True)
    else:
        print(f"[voices] ref_dir does not exist: {_ref_dir}", flush=True)

    return voices


@app.post("/tts")
def generate_tts(req: TtsRequest):
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ref_wav_path = _resolve_ref_wav(req.voice_id)
    use_no_ref = ref_wav_path is None

    use_caption = (
        _runtime.model_cfg.use_caption_condition
        and req.caption is not None
        and req.caption.strip() != ""
    )
    use_speaker = _runtime.model_cfg.use_speaker_condition and not use_no_ref

    # Adjust cfg scales: zero out unused conditioning branches
    cfg_scale_text = req.cfg_scale_text
    cfg_scale_caption = req.cfg_scale_caption if use_caption else 0.0
    cfg_scale_speaker = req.cfg_scale_speaker if use_speaker else 0.0

    try:
        result = _runtime.synthesize(
            SamplingRequest(
                text=req.text,
                caption=req.caption if use_caption else None,
                ref_wav=ref_wav_path,
                no_ref=use_no_ref,
                num_steps=req.num_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode=req.cfg_guidance_mode,
                seed=req.seed,
                seconds=req.seconds,
                trim_tail=req.trim_tail,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wav_bytes = _audio_to_wav_bytes(result.audio, result.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


# --------------------------------------------------------------------------- #
#  Startup                                                                     #
# --------------------------------------------------------------------------- #

def _load_runtime(args: argparse.Namespace) -> None:
    global _runtime, _ref_dir, _model_type

    _ref_dir = Path(args.ref_dir).expanduser().resolve()
    _ref_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")

    key = RuntimeKey(
        checkpoint=str(checkpoint),
        model_device=args.model_device,
        codec_repo=args.codec_repo,
        model_precision=args.model_precision,
        codec_device=args.codec_device or args.model_device,
        codec_precision=args.codec_precision or args.model_precision,
        compile_model=bool(args.compile_model),
        compile_dynamic=bool(args.compile_dynamic),
        reference_cache_size=int(args.reference_cache_size),
    )
    _runtime = InferenceRuntime.from_key(key)
    _model_type = "voicedesign" if _runtime.model_cfg.use_caption_condition else "base"
    print(f"[api_server] Runtime ready. model_type={_model_type}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS HTTP API server for SillyTavern.")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", default=None, help="Local checkpoint path (.pt/.safetensors).")
    ckpt_group.add_argument("--hf-checkpoint", default=None, help="Hugging Face repo id.")
    parser.add_argument("--ref-dir", default="references", help="Directory containing reference audio files.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--model-device", default=default_runtime_device())
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-device", default=None)
    parser.add_argument("--codec-precision", default=None)
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for core inference methods.",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic=True when --compile-model is enabled.",
    )
    parser.add_argument(
        "--reference-cache-size",
        type=int,
        default=16,
        help="Number of prepared reference latents to keep in the in-memory LRU cache.",
    )
    args = parser.parse_args()

    _load_runtime(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
