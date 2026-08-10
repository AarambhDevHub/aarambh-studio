#!/usr/bin/env python3
"""Create synthetic WAV clips and a tiny Phase 42 audio smoke fixture.

The fixture is intentionally synthetic — it verifies that the audio decode,
mel-spectrogram, frozen encoder, projector, fusion, and audio-QA eval pipeline
are wired correctly without downloading full audio datasets.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "audio_smoke"
AUDIO_DIR = OUT / "audio"
EVAL_OUT = ROOT / "data" / "eval" / "audio_qa_smoke"

# Tiny audio encoder config: 16 mel bins x 16 frames, 4x4 patches -> 16 patch tokens.
ENCODER_CONFIG = {
    "n_mels": 16,
    "max_frames": 16,
    "patch_mel": 4,
    "patch_time": 4,
    "audio_d_model": 16,
    "audio_layers": 1,
    "audio_heads": 2,
    "mlp_dim": 32,
    "norm_eps": 0.00001,
}

# Mel config matching the encoder's n_mels/max_frames. With sample_rate=16000,
# n_fft=256, hop_length=128, max_audio_seconds must yield max_frames=16:
#   total_samples = (16 - 1) * 128 + 256 = 2176 -> 2176/16000 = 0.136 s.
MEL_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 256,
    "hop_length": 128,
    "n_mels": 16,
    "fmin": 0.0,
    "fmax": 8000.0,
    "max_audio_seconds": 0.136,
    "power": 2.0,
    "log_eps": 0.0000000001,
}

# Tiny language model hidden width (matches configs/tiny_shakespeare).
LLM_HIDDEN = 384
PROJECTOR_HIDDEN_MULT = 1


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    data = bytearray()
    header: dict[str, object] = {}
    for name in sorted(tensors):
        tensor = np.ascontiguousarray(tensors[name], dtype=np.float32)
        start = len(data)
        payload = tensor.tobytes(order="C")
        data.extend(payload)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [start, start + len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)


def write_wav(path: Path, frequency: float, duration: float = 0.3) -> None:
    sample_rate = MEL_CONFIG["sample_rate"]
    sample_count = int(duration * sample_rate)
    samples = np.sin(
        2.0 * math.pi * frequency * np.arange(sample_count) / sample_rate
    ).astype(np.float32)
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def make_encoder_fixture() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio_encoder_tiny_config.json").write_text(
        json.dumps(ENCODER_CONFIG, indent=2) + "\n", encoding="utf-8"
    )
    rng = np.random.default_rng(42)
    width = ENCODER_CONFIG["audio_d_model"]
    mlp = ENCODER_CONFIG["mlp_dim"]
    patch_dim = ENCODER_CONFIG["patch_mel"] * ENCODER_CONFIG["patch_time"]
    num_patches = (
        ENCODER_CONFIG["n_mels"] // ENCODER_CONFIG["patch_mel"]
    ) * (ENCODER_CONFIG["max_frames"] // ENCODER_CONFIG["patch_time"])

    def randn(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    def ones(shape):
        return np.ones(shape, dtype=np.float32)

    tensors = {
        "patch_embed.weight": randn((width, patch_dim)),
        "class_embedding": randn((width,)),
        "position_embedding": randn((num_patches + 1, width), 0.01),
        "pre_norm.weight": ones((width,)),
        "pre_norm.bias": zeros((width,)),
        "post_norm.weight": ones((width,)),
        "post_norm.bias": zeros((width,)),
        "blocks.0.norm1.weight": ones((width,)),
        "blocks.0.norm1.bias": zeros((width,)),
        "blocks.0.norm2.weight": ones((width,)),
        "blocks.0.norm2.bias": zeros((width,)),
        "blocks.0.attn.q_proj.weight": randn((width, width)),
        "blocks.0.attn.q_proj.bias": zeros((width,)),
        "blocks.0.attn.k_proj.weight": randn((width, width)),
        "blocks.0.attn.k_proj.bias": zeros((width,)),
        "blocks.0.attn.v_proj.weight": randn((width, width)),
        "blocks.0.attn.v_proj.bias": zeros((width,)),
        "blocks.0.attn.out_proj.weight": randn((width, width)),
        "blocks.0.attn.out_proj.bias": zeros((width,)),
        "blocks.0.mlp.fc1.weight": randn((mlp, width)),
        "blocks.0.mlp.fc1.bias": zeros((mlp,)),
        "blocks.0.mlp.fc2.weight": randn((width, mlp)),
        "blocks.0.mlp.fc2.bias": zeros((width,)),
    }
    write_safetensors(OUT / "audio_encoder_tiny.safetensors", tensors)


def make_projector_fixture() -> None:
    rng = np.random.default_rng(7)
    audio_d_model = ENCODER_CONFIG["audio_d_model"]
    hidden = LLM_HIDDEN * PROJECTOR_HIDDEN_MULT

    def randn(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    tensors = {
        "fc1.weight": randn((hidden, audio_d_model)),
        "fc1.bias": zeros((hidden,)),
        "fc2.weight": randn((LLM_HIDDEN, hidden)),
        "fc2.bias": zeros((LLM_HIDDEN,)),
    }
    write_safetensors(OUT / "projector_init.safetensors", tensors)


def make_audio_clips_and_qa() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clips = [
        ("low_tone.wav", 220.0, "a low tone"),
        ("mid_tone.wav", 440.0, "a mid tone"),
        ("high_tone.wav", 880.0, "a high tone"),
        ("chord.wav", 660.0, "a chord"),
    ]
    for filename, frequency, _answer in clips:
        write_wav(AUDIO_DIR / filename, frequency)
    records = [
        {"audio": filename, "question": "What sound is this?", "answer": answer}
        for filename, _frequency, answer in clips
    ]
    (OUT / "audio_qa_smoke_4.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    eval_records = [
        {**record, "audio": f"../../audio_smoke/audio/{record['audio']}"}
        for record in records[:2]
    ]
    (EVAL_OUT / "data.jsonl").write_text(
        "\n".join(json.dumps(record) for record in eval_records) + "\n", encoding="utf-8"
    )
    print(f"audio smoke data: {OUT / 'audio_qa_smoke_4.jsonl'}")
    print(f"audio smoke eval: {EVAL_OUT / 'data.jsonl'}")
    print(f"audio smoke clips: {AUDIO_DIR}")


def main() -> None:
    make_encoder_fixture()
    make_projector_fixture()
    make_audio_clips_and_qa()


if __name__ == "__main__":
    main()
