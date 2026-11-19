#!/usr/bin/env bash
# Register a fine-tuned 4-bit GGUF with Ollama as "triage-<system>", then time it.
#   bash scripts/ollama_create.sh qwen3-0.6b       # uses models/qwen3-0.6b/*.gguf
# The model was trained on a plain prompt (no chat template), so the Modelfile keeps the
# template minimal and the evaluation calls Ollama in raw mode.
set -euo pipefail
cd "$(dirname "$0")/.."
SYSTEM="${1:?usage: ollama_create.sh <system, e.g. qwen3-0.6b>}"
GGUF="$(ls models/"$SYSTEM"/*.gguf | grep -i -E 'q4_k_m|Q4_K_M' | head -1 || true)"
GGUF="${GGUF:-$(ls models/"$SYSTEM"/*.gguf | head -1)}"
NAME="triage-$SYSTEM"
cat > "models/$SYSTEM/Modelfile" <<EOF
FROM ./$(basename "$GGUF")
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0
PARAMETER num_predict 48
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "\n###"
EOF
# A 16-bit GGUF (local export) is quantised to 4 bits by Ollama; a q4_k_m GGUF (Colab) is used as is.
QUANT=()
if ! echo "$GGUF" | grep -qi 'q4_k_m'; then QUANT=(--quantize q4_K_M); fi
(cd "models/$SYSTEM" && ollama create "$NAME" "${QUANT[@]}" -f Modelfile)
ollama list | grep "$NAME"
echo "next: OLLAMA_MODEL=$NAME python -m evals.latency --system ollama"
echo "      OLLAMA_MODEL=$NAME python -m evals.run --systems ollama --set test --limit 200"
