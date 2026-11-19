"""Merge a LoRA adapter into its base model and write a 16-bit GGUF, on a CPU, without Colab.

    python scripts/export_gguf_local.py smollm2-135m      # the original 135M adapter in adapter/
    python scripts/export_gguf_local.py qwen3-0.6b        # adapters/qwen3-0.6b (fallback if Colab's export failed)
    bash scripts/ollama_create.sh <system>                # then Ollama quantises it to 4 bits (q4_K_M)

Uses the conversion script of llama.cpp (shallow clone into .cache/llama.cpp, pinned tag).
Needs: pip install gguf sentencepiece protobuf  (or uv pip install ...).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LLAMA_CPP_TAG = "b6000"


def main() -> None:
    system = sys.argv[1] if len(sys.argv) > 1 else "smollm2-135m"
    from triage.config import get_settings, qwen_model
    from triage.model import load_model, load_tokenizer

    if system == "smollm2-135m":
        settings = get_settings()
        base, adapter = settings.base_model, settings.adapter_dir
    else:
        base, adapter = qwen_model(system)
    merged = ROOT / ".cache" / f"merged-{system}"
    out = ROOT / "models" / system
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(base, adapter, merge=True)  # float32 on CPU, adapter folded into the weights
    tokenizer = load_tokenizer(base)
    model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)

    llama = ROOT / ".cache" / "llama.cpp"
    if not llama.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", LLAMA_CPP_TAG,
                        "https://github.com/ggml-org/llama.cpp", str(llama)], check=True)
    target = out / f"{system}-f16.gguf"
    import os

    env = {**os.environ, "PYTHONPATH": str(llama / "gguf-py")}  # the gguf package that matches the script
    subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged),
                    "--outtype", "f16", "--outfile", str(target)], check=True, env=env)
    print("wrote", target.relative_to(ROOT), f"{target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
