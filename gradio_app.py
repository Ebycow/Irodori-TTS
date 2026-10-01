#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np

from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    download_hf_checkpoint,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
)
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path

MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 8
LOCAL_MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_CHUNK_MAX_CHARS = 40
DEFAULT_CHUNK_GAP_MS = 250

_SENTENCE_END_CHARS = "。．！？!?…‥\n"
_SENTENCE_TRAIL_CHARS = "。．！？!?…‥」』）)】〉》\"'”’〜ー~"
_CLAUSE_END_CHARS = "、，,；;：:　 "


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *(
                path
                for path in Path(".").glob("**/checkpoint_*.safetensors")
                if not is_speaker_inversion_safetensors_path(path)
            ),
        ]
    )
    if not candidates:
        local = _local_model_checkpoint("Aratako/Irodori-TTS-v4-Large")
        return str(local) if local is not None else "Aratako/Irodori-TTS-v4-Large"
    return str(candidates[-1])


def _local_model_checkpoint(source: str) -> Path | None:
    """Map a path or HF repo id to a model.safetensors under models/ if present."""
    path = Path(source).expanduser()
    if path.is_dir() and (path / "model.safetensors").is_file():
        return path / "model.safetensors"
    local = LOCAL_MODELS_DIR / source.strip("/").split("/")[-1] / "model.safetensors"
    return local if local.is_file() else None


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}:
        return None
    return text


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


def _is_trailing_symbol(ch: str) -> bool:
    """Chars that stay attached to the preceding sentence end (closers, emoji, ZWJ, VS)."""
    if ch in _SENTENCE_TRAIL_CHARS:
        return True
    if ch in {"\u200d", "\ufe0f", "\ufe0e"}:
        return True
    return unicodedata.category(ch) in {"So", "Sk", "Mn", "Cf"}


def _split_at(text: str, end_chars: str, trailing: bool) -> list[str]:
    pieces: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in end_chars:
            j = i + 1
            while trailing and j < n and text[j] != "\n" and _is_trailing_symbol(text[j]):
                j += 1
            pieces.append(text[start:j])
            start = j
            i = j
        else:
            i += 1
    if start < n:
        pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split long text at sentence boundaries into chunks of at most ~max_chars characters.

    Sentences longer than max_chars are further split at clause punctuation, and as a
    last resort hard-split by length.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if text == "":
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [re.sub(r"\s*\n\s*", " ", text).strip()]

    pieces: list[str] = []
    for sentence in _split_at(text, _SENTENCE_END_CHARS, trailing=True):
        if len(sentence.strip()) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in _split_at(sentence, _CLAUSE_END_CHARS, trailing=False):
            if len(clause) <= max_chars:
                pieces.append(clause)
            else:
                pieces.extend(clause[k : k + max_chars] for k in range(0, len(clause), max_chars))

    chunks = [re.sub(r"\s*\n\s*", " ", chunk).strip() for chunk in _pack(pieces, max_chars)]
    return [chunk for chunk in chunks if chunk]


def _chunk_table(chunks: list[str]) -> list[list[object]]:
    return [[i + 1, len(chunk), chunk] for i, chunk in enumerate(chunks)]


def _chunk_choices(chunks: list[str]) -> list[str]:
    return [
        f"{i + 1}: {chunk[:30]}{'…' if len(chunk) > 30 else ''}" for i, chunk in enumerate(chunks)
    ]


def _preview_split(text: str, split_enabled: bool, max_chars: float) -> tuple[object, str]:
    chunks = split_text_into_chunks(text, int(max_chars) if split_enabled else 0)
    summary = f"{len(chunks)} chunk(s), {sum(len(c) for c in chunks)} chars"
    return _chunk_table(chunks), summary


def _select_chunk_audio(
    chunk_audios: list[list[tuple[int, np.ndarray]]] | None,
    chunk_label: str | None,
    candidate: str | None,
) -> object:
    if not chunk_audios or not chunk_label:
        return None
    chunk_idx = int(str(chunk_label).split(":", 1)[0]) - 1
    cand_idx = int(candidate or 1) - 1
    if not 0 <= chunk_idx < len(chunk_audios):
        return None
    candidates = chunk_audios[chunk_idx]
    if not 0 <= cand_idx < len(candidates):
        return None
    return candidates[cand_idx]


def _coerce_gradio_file_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def _resolve_ref_wavs(uploaded_audio: object) -> list[str]:
    if uploaded_audio is None:
        return []
    values = uploaded_audio if isinstance(uploaded_audio, (list, tuple)) else [uploaded_audio]
    paths = [_coerce_gradio_file_path(value) for value in values]
    return [path for path in paths if path is not None]


def _resolve_speaker_embedding(
    uploaded_embedding: object,
    speaker_embedding_path_raw: str | None,
) -> str | None:
    uploaded_path = _coerce_gradio_file_path(uploaded_embedding)
    raw_path = None
    if speaker_embedding_path_raw is not None and str(speaker_embedding_path_raw).strip():
        raw_path = str(speaker_embedding_path_raw).strip()
    if uploaded_path is not None and raw_path is not None:
        raise ValueError("Use either speaker embedding upload or speaker embedding path, not both.")
    return uploaded_path if uploaded_path is not None else raw_path


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    local = _local_model_checkpoint(checkpoint)
    if local is not None:
        print(f"[gradio] checkpoint: {checkpoint} -> {local} (local)", flush=True)
        return str(local)

    resolved = download_hf_checkpoint(checkpoint)
    print(f"[gradio] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> str:
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    status_text = (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}"
    )
    return status_text


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    text: str,
    uploaded_audio: object,
    uploaded_speaker_embedding: object,
    speaker_embedding_path_raw: str,
    num_steps: str | None,
    num_candidates: int,
    seed_raw: str,
    seconds_raw: str,
    duration_scale: float,
    t_schedule_mode: str,
    sway_coeff: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    lora_adapter_raw: str,
    split_enabled: bool,
    chunk_max_chars: float,
    chunk_gap_ms: float,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
    )

    if str(text).strip() == "":
        raise ValueError("text is required.")
    chunks = split_text_into_chunks(text, int(chunk_max_chars) if split_enabled else 0)
    if not chunks:
        raise ValueError("text is required.")
    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    parsed_num_steps = _parse_optional_int(num_steps, "num_steps")
    if parsed_num_steps is not None and parsed_num_steps < 1:
        raise ValueError("num_steps must be >= 1 or blank.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)
    if manual_seconds is not None and len(chunks) > 1:
        stdout_log("[gradio] warning: manual seconds is ignored when text is split into chunks")
        manual_seconds = None

    ref_wavs = _resolve_ref_wavs(uploaded_audio)
    speaker_embedding = _resolve_speaker_embedding(
        uploaded_embedding=uploaded_speaker_embedding,
        speaker_embedding_path_raw=speaker_embedding_path_raw,
    )
    if ref_wavs and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker embedding are mutually exclusive.")
    no_ref = not ref_wavs and speaker_embedding is None
    ref_normalize_db = -16.0
    ref_ensure_max = True

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "mode={} schedule={} sway_coeff={} seconds={} duration_scale={} steps={} seed={} no_ref={} candidates={}"
        ).format(
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            cfg_guidance_mode,
            t_schedule_mode,
            sway_coeff,
            "auto" if manual_seconds is None else manual_seconds,
            duration_scale,
            num_steps,
            "random" if seed is None else seed,
            no_ref,
            requested_candidates,
        )
    )
    if speaker_embedding is not None:
        stdout_log(f"[gradio] speaker_embedding: {speaker_embedding}")
    elif ref_wavs:
        stdout_log(f"[gradio] reference clips: {len(ref_wavs)}")

    stdout_log(f"[gradio] text split into {len(chunks)} chunk(s)")

    def synthesize_chunk(chunk_text: str, chunk_seed: int | None) -> object:
        return runtime.synthesize(
            SamplingRequest(
                text=chunk_text,
                ref_wav=None,
                ref_wavs=ref_wavs or None,
                ref_latent=None,
                ref_embed=speaker_embedding,
                no_ref=bool(no_ref),
                ref_normalize_db=ref_normalize_db,
                ref_ensure_max=bool(ref_ensure_max),
                num_candidates=requested_candidates,
                decode_mode="sequential",
                seconds=manual_seconds,
                duration_scale=float(duration_scale),
                max_ref_seconds=None,
                max_text_len=None,
                num_steps=parsed_num_steps,
                seed=chunk_seed,
                cfg_guidance_mode=str(cfg_guidance_mode),
                cfg_scale_text=float(cfg_scale_text),
                cfg_scale_speaker=float(cfg_scale_speaker),
                cfg_scale=cfg_scale,
                cfg_min_t=float(cfg_min_t),
                cfg_max_t=float(cfg_max_t),
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                context_kv_cache=bool(context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers,
                t_schedule_mode=str(t_schedule_mode),
                sway_coeff=float(sway_coeff),
                trim_tail=True,
                lora_adapter=lora_adapter,
            ),
            log_fn=stdout_log,
        )

    # All chunks share one seed so that the voice stays consistent across chunks.
    chunk_seed = None if seed is None else int(seed)
    sample_rate: int | None = None
    chunk_audios: list[list[tuple[int, np.ndarray]]] = []
    timing_blocks: list[str] = []
    chunk_messages: list[str] = []
    for chunk_idx, chunk_text in enumerate(chunks):
        stdout_log(f"[gradio] chunk {chunk_idx + 1}/{len(chunks)}: {chunk_text}")
        result = synthesize_chunk(chunk_text, chunk_seed)
        if chunk_seed is None:
            chunk_seed = int(result.used_seed)
        sample_rate = int(result.sample_rate)
        chunk_audios.append([(sample_rate, a.float().squeeze(0).numpy()) for a in result.audios])
        prefix = f"[chunk {chunk_idx + 1}/{len(chunks)}] " if len(chunks) > 1 else ""
        chunk_messages.extend(f"{prefix}{msg}" for msg in result.messages)
        timing_blocks.append(
            f"{prefix}\n{_format_timings(result.stage_timings, result.total_to_decode)}"
            if prefix
            else _format_timings(result.stage_timings, result.total_to_decode)
        )

    assert sample_rate is not None
    gap = np.zeros(max(0, int(sample_rate * float(chunk_gap_ms) / 1000.0)), dtype=np.float32)
    audio_tensors: list[tuple[int, np.ndarray]] = []
    for cand_idx in range(requested_candidates):
        parts: list[np.ndarray] = []
        for candidates in chunk_audios:
            if cand_idx >= len(candidates):
                continue
            if parts and gap.size > 0:
                parts.append(gap)
            parts.append(np.asarray(candidates[cand_idx][1], dtype=np.float32))
        if parts:
            audio_tensors.append((sample_rate, np.concatenate(parts)))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {chunk_seed}",
        f"chunks: {len(chunks)}",
        f"candidates: {len(audio_tensors)}",
        *chunk_messages,
    ]
    detail_text = "\n".join(detail_lines)
    timing_text = "\n".join(timing_blocks)
    stdout_log(f"[gradio] generated {len(audio_tensors)} candidates from {len(chunks)} chunk(s)")

    audio_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(audio_tensors):
            audio_updates.append(gr.update(value=audio_tensors[i], visible=True))
        else:
            audio_updates.append(gr.update(value=None, visible=False))

    choices = _chunk_choices(chunks)
    first_chunk_audio = chunk_audios[0][0] if chunk_audios and chunk_audios[0] else None
    return (
        *audio_updates,
        detail_text,
        timing_text,
        _chunk_table(chunks),
        f"{len(chunks)} chunk(s), {sum(len(c) for c in chunks)} chars",
        chunk_audios,
        gr.update(choices=choices, value=choices[0]),
        gr.update(choices=[str(i + 1) for i in range(len(audio_tensors))], value="1"),
        first_chunk_audio,
    )


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Gradio") as demo:
        gr.Markdown("# Irodori-TTS Inference (Cached Runtime)")
        gr.Markdown(
            "Reference-audio cloning / Speaker Inversion UI. "
            "Irodori-TTS-v4.1-Small is used by default; unchanged settings reuse the cached runtime."
        )

        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
            )
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[0],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[0],
                scale=1,
            )

        with gr.Row():
            load_model_btn = gr.Button("Load Model")
            clear_cache_btn = gr.Button("Unload Model")
            clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

        with gr.Column():
            text = gr.Textbox(
                label="Text",
                lines=6,
                elem_id="irodori-text-input",
            )
            build_emoji_palette(text, open=False)
        with gr.Accordion("Long Text Splitting", open=True):
            gr.Markdown(
                "Long text is split at sentence boundaries (。！？ / newlines), generated chunk by "
                "chunk with the same seed, and concatenated. Manual Seconds is ignored when split."
            )
            with gr.Row():
                split_enabled = gr.Checkbox(label="Split Long Text", value=True)
                chunk_max_chars = gr.Slider(
                    label="Max Chars per Chunk",
                    minimum=10,
                    maximum=200,
                    value=DEFAULT_CHUNK_MAX_CHARS,
                    step=5,
                )
                chunk_gap_ms = gr.Slider(
                    label="Silence Between Chunks (ms)",
                    minimum=0,
                    maximum=1500,
                    value=DEFAULT_CHUNK_GAP_MS,
                    step=10,
                )
                preview_split_btn = gr.Button("Preview Split")
            split_summary = gr.Textbox(label="Split Summary", interactive=False)
            chunk_table = gr.Dataframe(
                headers=["#", "Chars", "Text"],
                datatype=["number", "number", "str"],
                column_widths=["6%", "8%", "86%"],
                wrap=True,
                interactive=False,
                label="Chunks",
            )
        with gr.Tabs():
            with gr.Tab("Reference Audio"):
                gr.Markdown(
                    "**Long-reference tip:** Upload multiple clean, shorter clips from the same "
                    "speaker and arrange them in the desired order. This matches v4-Small "
                    "training. A single uninterrupted long recording is accepted but has not "
                    "been evaluated."
                )
                uploaded_audio = gr.File(
                    label=("Reference Audio Uploads (optional; concatenated in displayed order)"),
                    type="filepath",
                    file_count="multiple",
                    file_types=["audio"],
                    allow_reordering=True,
                )
            with gr.Tab("Speaker Embedding"):
                with gr.Row():
                    uploaded_speaker_embedding = gr.File(
                        label="Speaker Embedding Upload (.speaker.safetensors, optional)",
                        type="filepath",
                        file_count="single",
                        scale=1,
                    )
                    speaker_embedding_path_raw = gr.Textbox(
                        label="Speaker Embedding Path (.speaker.safetensors, optional)",
                        value="",
                        scale=1,
                    )

        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Textbox(
                    label="Num Steps",
                    value="",
                    placeholder="Auto (RF: 40, MeanFlow: 4)",
                    info="Blank: model default (RF: 40, MeanFlow: 4).",
                )
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
                )
                seed_raw = gr.Textbox(label="Seed (blank=random)", value="")
                seconds_raw = gr.Textbox(label="Seconds (blank=auto)", value="")
                duration_scale = gr.Slider(
                    label="Duration Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.01,
                )

            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="Time Schedule",
                    choices=["linear", "sway"],
                    value="linear",
                )
                sway_coeff = gr.Slider(
                    label="Sway Coeff",
                    minimum=-1.0,
                    maximum=1.5,
                    value=-1.0,
                    step=0.1,
                    interactive=False,
                )

            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced (Optional)", open=False):
            cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
            with gr.Row():
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="Speaker KV Max Layers (optional)", value=""
                )
            lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory (optional)", value="")

        generate_btn = gr.Button("Generate", variant="primary")

        out_audios: list[gr.Audio] = []
        num_rows = (
            MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
        ) // GRADIO_AUDIO_COLS_PER_ROW
        with gr.Column():
            for row_idx in range(num_rows):
                with gr.Row():
                    for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                        i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                        if i >= MAX_GRADIO_CANDIDATES:
                            break
                        out_audios.append(
                            gr.Audio(
                                label=f"Generated Audio {i + 1}",
                                type="numpy",
                                interactive=False,
                                visible=(i == 0),
                                min_width=160,
                            )
                        )
        with gr.Accordion("Chunk Audio", open=False):
            chunk_audios_state = gr.State([])
            with gr.Row():
                chunk_select = gr.Dropdown(label="Chunk", choices=[], scale=3)
                chunk_candidate = gr.Dropdown(label="Candidate", choices=["1"], value="1", scale=1)
            chunk_audio = gr.Audio(label="Chunk Audio", type="numpy", interactive=False)
        out_log = gr.Textbox(label="Run Log", lines=8)
        out_timing = gr.Textbox(label="Timing", lines=8)

        generate_btn.click(
            _run_generation,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                text,
                uploaded_audio,
                uploaded_speaker_embedding,
                speaker_embedding_path_raw,
                num_steps,
                num_candidates,
                seed_raw,
                seconds_raw,
                duration_scale,
                t_schedule_mode,
                sway_coeff,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
                lora_adapter_raw,
                split_enabled,
                chunk_max_chars,
                chunk_gap_ms,
            ],
            outputs=[
                *out_audios,
                out_log,
                out_timing,
                chunk_table,
                split_summary,
                chunk_audios_state,
                chunk_select,
                chunk_candidate,
                chunk_audio,
            ],
        )
        preview_split_btn.click(
            _preview_split,
            inputs=[text, split_enabled, chunk_max_chars],
            outputs=[chunk_table, split_summary],
        )
        for trigger in (chunk_select.input, chunk_candidate.input):
            trigger(
                _select_chunk_audio,
                inputs=[chunk_audios_state, chunk_select, chunk_candidate],
                outputs=[chunk_audio],
            )
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )
        t_schedule_mode.change(
            _on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff]
        )

        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
            ],
            outputs=[clear_cache_msg],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for Irodori-TTS with cached runtime.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        css=EMOJI_PALETTE_CSS,
    )


if __name__ == "__main__":
    main()
