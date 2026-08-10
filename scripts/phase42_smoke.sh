#!/usr/bin/env bash
set -euo pipefail

BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
DOCUMENT_DIR=${AUDIO_DOCUMENT_MIGRATION_DIR:-checkpoints/document_smoke}
AUDIO_DIR=${AUDIO_CHECKPOINT_DIR:-checkpoints/audio_smoke}
ADAPTER_DIR=${AUDIO_ADAPTER_DIR:-adapters/audio_qa_smoke}
MERGED_DIR=${AUDIO_MERGED_DIR:-checkpoints/audio_qa_smoke_merged}
SCORECARD=${AUDIO_SCORECARD:-artifacts/phase42_audio_smoke.json}

[[ -x "$BIN" ]] || {
  echo "missing executable Phase 42 binary: $BIN" >&2
  exit 2
}
[[ -s "$DOCUMENT_DIR/model.safetensors" ]] || {
  echo "missing Phase 42 base checkpoint: $DOCUMENT_DIR/model.safetensors (run phase36_smoke.sh first)" >&2
  exit 2
}
[[ -s "$DOCUMENT_DIR/tokenizer.json" ]] || {
  echo "missing Phase 42 base tokenizer: $DOCUMENT_DIR/tokenizer.json (run phase36_smoke.sh first)" >&2
  exit 2
}

python3 scripts/phase42_make_audio_smoke_fixture.py
mkdir -p "$AUDIO_DIR" "$ADAPTER_DIR" "$MERGED_DIR" "$(dirname "$SCORECARD")"

if [[ ! -s "$AUDIO_DIR/model.safetensors" || ! -s "$AUDIO_DIR/tokenizer.json" ]]; then
  "$BIN" convert \
    --config configs/document_qa_smoke.toml \
    --input "$DOCUMENT_DIR/model.safetensors" \
    --output "$AUDIO_DIR/model.safetensors" \
    --tokenizer "$DOCUMENT_DIR/tokenizer.json" \
    --output-tokenizer "$AUDIO_DIR/tokenizer.json" \
    --upgrade-audio-vocab
fi

"$BIN" finetune audio-dora \
  --config configs/audio_qa_smoke.toml \
  --base "$AUDIO_DIR/model.safetensors" \
  --tokenizer "$AUDIO_DIR/tokenizer.json" \
  --data data/audio_smoke/audio_qa_smoke_4.jsonl \
  --output "$ADAPTER_DIR" \
  --projector data/audio_smoke/projector_init.safetensors \
  --lora-rank 4 \
  --batch-size 1 \
  --max-steps 2 \
  --log-every-n-steps 1 \
  --save-every-n-steps 0

for artifact in adapter.safetensors projector.safetensors adapter_config.json audio_adapter_config.json; do
  [[ -s "$ADAPTER_DIR/$artifact" ]] || {
    echo "missing Phase 42 training artifact: $ADAPTER_DIR/$artifact" >&2
    exit 2
  }
done

"$BIN" finetune merge \
  --config configs/audio_qa_smoke.toml \
  --base "$AUDIO_DIR/model.safetensors" \
  --adapter "$ADAPTER_DIR" \
  --method dora \
  --output "$MERGED_DIR"

"$BIN" infer \
  --config configs/audio_qa_smoke_infer.toml \
  --model "$MERGED_DIR/model.safetensors" \
  --tokenizer "$AUDIO_DIR/tokenizer.json" \
  --audio data/audio_smoke/audio/mid_tone.wav \
  --prompt "What sound is this?" \
  --max-tokens 2 \
  --greedy \
  --safety none

"$BIN" eval \
  --config configs/audio_qa_smoke_infer.toml \
  --model "$MERGED_DIR/model.safetensors" \
  --tokenizer "$AUDIO_DIR/tokenizer.json" \
  --tasks audio-qa-smoke \
  --data-dir data/eval \
  --max-examples 2 \
  --max-new-tokens 2 \
  --out "$SCORECARD"

[[ -s "$SCORECARD" ]] || {
  echo "missing Phase 42 scorecard: $SCORECARD" >&2
  exit 2
}
echo "Phase 42 audio smoke completed: $SCORECARD"
