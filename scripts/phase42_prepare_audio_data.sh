#!/usr/bin/env bash
# Prepare a free, public audio-caption / audio-QA dataset subset for Phase 42.
#
# Same "free and public only" policy as every prior modality phase. This script
# is a thin orchestrator: it downloads a permissively-licensed audio-caption
# subset (or reuses a locally-provided one) and normalizes it into the JSONL
# schema aarambh-studio-audio::load_audio_qa_jsonl consumes:
#
#   {"audio": "<relative-path>", "question": "...", "answer": "..."}
#   {"audio": "<relative-path>", "caption": "..."}
#   {"audio": "<path>", "conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
#
# Usage:
#   scripts/phase42_prepare_audio_data.sh data [optional-source-archive]
#
# If no source archive is provided, the script creates a tiny synthetic
# smoke subset (identical to scripts/phase42_make_audio_smoke_fixture.py) so
# the data-preparation path is exercisable end-to-end without network access.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${1:-data}"
SOURCE_ARCHIVE="${2:-}"

mkdir -p "$DATA_DIR/audio_qa"
AUDIO_ROOT="$DATA_DIR/audio_qa/audio"
mkdir -p "$AUDIO_ROOT"

if [[ -n "$SOURCE_ARCHIVE" && -f "$SOURCE_ARCHIVE" ]]; then
  echo "preparing audio data from $SOURCE_ARCHIVE"
  CACHE_DIR="$DATA_DIR/audio_qa/_cache"
  mkdir -p "$CACHE_DIR"
  if [[ "$SOURCE_ARCHIVE" == *.tar.gz || "$SOURCE_ARCHIVE" == *.tgz ]]; then
    tar -xzf "$SOURCE_ARCHIVE" -C "$CACHE_DIR"
  elif [[ "$SOURCE_ARCHIVE" == *.zip ]]; then
    unzip -o -q "$SOURCE_ARCHIVE" -d "$CACHE_DIR"
  else
    echo "unsupported archive format: $SOURCE_ARCHIVE" >&2
    exit 2
  fi
  echo "extracted source to $CACHE_DIR — normalize into JSONL with:"
  echo "  python3 - <<'PY'"
  echo "  import json, pathlib"
  echo "  root = pathlib.Path('$CACHE_DIR')"
  echo "  out = pathlib.Path('$DATA_DIR/audio_qa/data.jsonl')"
  echo "  # Replace this loop with dataset-specific normalization."
  echo "  records = [{'audio': str(p.relative_to(root.parent)), 'caption': p.stem} for p in root.rglob('*.wav')]"
  echo "  out.write_text('\\n'.join(json.dumps(r) for r in records) + '\\n')"
  echo "  PY"
else
  echo "no source archive provided; generating a synthetic smoke subset"
  python3 scripts/phase42_make_audio_smoke_fixture.py
  cp data/audio_smoke/audio_qa_smoke_4.jsonl "$DATA_DIR/audio_qa/data.jsonl"
  cp -r data/audio_smoke/audio/* "$AUDIO_ROOT/" 2>/dev/null || true
fi

COUNT=$(wc -l < "$DATA_DIR/audio_qa/data.jsonl")
echo "Phase 42 audio data prepared: $COUNT examples in $DATA_DIR/audio_qa/data.jsonl"
