# Phase 42 — Audio Modality

> v4.0.0-alpha.2 · `aarambh-studio-audio` (new crate) · depends on v2 §24–25 (vision fusion pattern), v1 §7 (thinking engine)

Phase 42 adds a fourth input sense — **audio** — following the exact same
frozen-encoder-plus-trainable-projector pattern v2 §24 established for vision
and v3 §35–36 reused for video and documents. A frozen, pretrained audio
spectrogram transformer converts a log-mel spectrogram into a grid of patch
embeddings; a small trainable projector maps those into the decoder's `d_model`
space; the result is spliced into the token sequence at the `<audio>` special
token position. Nothing about the decoder, the thinking engine, grammar-
constrained tool calling, or long-horizon tool chains changes — audio is just
another sense feeding the same fusion mechanism.

## Mechanism

```
Raw audio waveform (WAV PCM 8/16/24/32-bit, mono or stereo)
     │
     ▼
Pure-Rust WAV decode + linear resample to target sample rate
(no Python audio ML tooling, no system-library FFI)
     │
     ▼
Mel-spectrogram extraction (Hann window + radix-2 Cooley-Tukey FFT
+ triangular mel filterbank + log-mel + per-frame normalization)
     │
     ▼
FrozenAudioEncoder (pretrained, loaded as SafeTensors via candle-core —
same loading path as CLIP; AST-style patchify + transformer blocks)
     │
     ▼
Projector MLP (trainable): audio_d_model -> hidden -> llm_d_model
     │
     ▼
N "audio tokens" in llm_d_model space
     │
     ▼
Spliced into the input sequence at the <audio> special token position
     │
     ▼
...the rest of the decoder, completely unmodified...
```

### Two-stage training

Identical structure to v2 §25's vision recipe:

1. **Projector-only stage.** Everything else frozen; the projector trains alone
   on audio-captioning-style data, learning to map audio embeddings into a space
   the (frozen) decoder can already interpret reasonably.
2. **Instruction-tuning stage.** The projector continues training alongside a
   DoRA-adapted (v2 §23) LLM on open-ended audio-QA data. `finetune audio-dora`
   runs this stage; `--freeze-projector` selects projector-only training.

## New crate: `aarambh-studio-audio`

| Module | Role |
|---|---|
| `encoder.rs` | `AudioEncoderConfig` + `FrozenAudioEncoder` — AST-style transformer over mel-spectrogram patches, `load_pretrained` via `VarBuilder::from_mmaped_safetensors` |
| `preprocess.rs` | `MelSpectrogramConfig` + `AudioPreprocessor` — pure-Rust WAV decode, resampling, Hann window, radix-2 FFT, mel filterbank, log-mel normalization (zero new dependencies) |
| `projector.rs` | `AudioProjectorConfig` + `AudioProjector` — two-layer GELU MLP mirroring `VisionProjector` exactly |
| `fusion.rs` | `interleave_audio_tokens()` — generalizes `interleave_image_tokens` to the shared modal-token-splicing pattern |
| `instruct_data.rs` | `AudioQaExample` + `load_audio_qa_jsonl` — caption, QA, and LLaVA-style conversation records |

## Tokenizer

Two new reserved special tokens, IDs 15 and 16, following the exact pattern of
v2's `<image>`/`<image_end>` and v3's `<video>`/`<document>` tokens:

| Token | String | ID |
|---|---|---|
| Audio placeholder | `<audio>` | 15 |
| Audio prefix boundary | `<audio_end>` | 16 |

`AUDIO_SPECIAL_TOKENS` (17 entries) is a strict superset of the Phase 36
document table. `BpeTokenizer::validate_audio_special_tokens()` checks a
checkpoint's tokenizer carries the audio tokens at their required IDs, and
`upgraded_for_audio()` migrates a document-capable tokenizer to the audio layout
(shifting learned token IDs ≥ 15 by two), exactly as `upgraded_for_document`
did for video → document.

`convert --upgrade-audio-vocab` applies the migration to a SafeTensors
checkpoint and its tokenizer together (`VocabularyExpansion { insertion_id: 15,
source_ids: [IMAGE_ID, IMAGE_END_ID] }`), copying the canonical placeholder
pair as initialization for the two new embedding rows.

## Configuration

Audio is configured under the existing `[vision]` block (the multimodal config
block document and video already share), as `[vision.audio]`:

```toml
[vision]
mode = "vlm_instruction"
base_model_path = "checkpoints/audio_smoke/model.safetensors"
projector_path = "data/audio_smoke/projector_init.safetensors"
projector_hidden_mult = 1
max_samples = 4

[vision.audio]
audio_root = "data/audio_smoke/audio"
encoder_config_path = "data/audio_smoke/audio_encoder_tiny_config.json"
encoder_weights_path = "data/audio_smoke/audio_encoder_tiny.safetensors"
encoder_batch_size = 2
feature_cache_entries = 4

[vision.audio.mel]
sample_rate = 16000
n_fft = 256
hop_length = 128
n_mels = 16
fmin = 0.0
fmax = 8000.0
max_audio_seconds = 0.136
power = 2.0
log_eps = 0.0000000001
```

See `configs/audio_qa_smoke.toml` and `configs/audio_qa_smoke_infer.toml` for
ready-to-use smoke recipes.

## CLI

```text
aarambh-studio infer --audio <path.wav> --prompt "What sound is this?"
aarambh-studio finetune audio-dora   --config configs/audio_qa_smoke.toml ...
aarambh-studio finetune audio-qdora  --config configs/audio_qa_smoke.toml ...
aarambh-studio convert --upgrade-audio-vocab ...
aarambh-studio eval --tasks audio-qa-smoke ...
```

`--audio` conflicts with `--image`, `--video`, and `--document` (one modality
per inference call, the same discipline every prior modality flag holds).

## Tests

The Phase 42 proof obligations (from `ROADMAP_V4.md`):

| Test | Location | Proves |
|---|---|---|
| `frozen_audio_encoder_never_receives_gradients` | `aarambh-studio-audio` | detached encoder output blocks gradients from reaching encoder parameters while the projector still trains |
| `projector_pretrain_stage_trains_only_projector_weights` | `aarambh-studio-audio` | the projector-only stage's structural freeze guarantee holds |
| `audio_token_fusion_produces_expected_sequence_length` | `aarambh-studio-audio` | `interleave_audio_tokens` produces `seq - 1 + audio_tokens` length |
| `thinking_controller_behaves_identically_after_audio_context` | `aarambh-studio-audio` | fused audio-context embeddings are a well-formed, finite, contiguous sequence the decoder observes identically to text |
| `audio_table_is_strict_superset_of_document_table` | `aarambh-studio-tokenizer` | the audio token table extends the document table by exactly two entries |
| `audio_token_ids_are_contiguous_after_document_tokens` | `aarambh-studio-tokenizer` | audio IDs 15/16 immediately follow the document table |

Plus unit tests for WAV decode (PCM 16-bit + 32-bit float stereo), FFT frequency
recovery, mel frame counting, projector output width, fusion placeholder
validation, and JSONL parsing (caption + QA + conversation formats).

### Smoke test

```sh
scripts/phase42_smoke.sh
```

Generates a synthetic audio fixture (sine-wave WAVs + a tiny random-init audio
encoder + projector), upgrades a document-capable checkpoint to the audio
vocabulary, runs a two-step `finetune audio-dora`, merges, runs
`infer --audio` on a held-out clip, and scores `audio-qa-smoke`. Requires the
Phase 36 document smoke checkpoint as its base (the same prerequisite chain
phase35 → phase36 → phase42 that vision → video → document → audio follows).

## Composability

Because fusion happens before the decoder sees the sequence, nothing about the
thinking engine (v1 §7), grammar-constrained tool calling (v2 §30), or
long-horizon tool chains (v3 §46) needs to change — a `ücklich` block or a tool
call generated after audio tokens behaves identically to one generated after
text-only or image-only context. This is the same composability guarantee every
prior modality addition has held, and the
`thinking_controller_behaves_identically_after_audio_context` test pins it.

## Scope and boundaries

- Audio **understanding** only (the model can be asked about audio it's given),
  not audio **generation** — aarambh-studio does not produce audio output. That
  capability belongs to a separate project entirely and is intentionally out of
  scope, the same boundary v2 §24 drew for vision.
- WAV PCM (8/16/24/32-bit signed and unsigned, plus 32/64-bit float) is decoded
  in pure Rust. MP3/FLAC/Ogg decode is future work, following the same
  "visual-only H.264 MP4" boundary v3 §35 set for video containers.
- Mel-spectrogram extraction uses a from-scratch Hann window, radix-2 FFT, and
  triangular mel filterbank — no `rustfft` or audio-DSP dependency, consistent
  with the project's "from first principles" dependency discipline.
- The frozen encoder reuses the candle fallback attention kernel. CUDA flash
  audio kernels are future work; the mechanism and fusion are in place.
- Self-learning (`SELF_LEARNING_V4.md`) is transparent to audio: online GRPO
  operates on token log-probabilities and has no dependency on the input
  modality. Audio self-learning is not wired into the `--self-learn` CLI flag
  (which remains text/image only, the same boundary video and document hold).
