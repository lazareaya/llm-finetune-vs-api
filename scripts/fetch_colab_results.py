"""Bring the Colab run back to this machine, from the Hugging Face repos the notebook pushed.

    python scripts/fetch_colab_results.py <hf_user>/qwen3-0.6b-support-triage-lora [--no-gguf]

- merges eval/report_entries.json into evals/report.json and copies the raw outputs
  into evals/outputs/ (used by the judge and the tables);
- downloads the adapter into adapters/<system>/;
- downloads the 4-bit GGUF from <hf_user>/<system>-support-triage-GGUF into models/<system>/.
Public repos need no token.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="e.g. yourname/qwen3-0.6b-support-triage-lora")
    parser.add_argument("--no-gguf", action="store_true")
    args = parser.parse_args()

    system = args.repo_id.split("/")[1].removesuffix("-support-triage-lora")  # qwen3-0.6b
    local = Path(snapshot_download(args.repo_id))

    entries = json.loads((local / "eval" / "report_entries.json").read_text())
    report_path = ROOT / "evals" / "report.json"
    report = json.loads(report_path.read_text())
    report.update(entries)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("merged into evals/report.json:", ", ".join(entries))

    outputs = ROOT / "evals" / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    for f in (local / "eval" / "outputs").glob("*.jsonl"):
        shutil.copy(f, outputs / f.name)

    adapter_dir = ROOT / "adapters" / system
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        shutil.copy(local / name, adapter_dir / name)
    shutil.copy(local / "eval" / "training_log.json", adapter_dir / "training_log.json")
    print("adapter in", adapter_dir.relative_to(ROOT))

    if not args.no_gguf:
        gguf_repo = args.repo_id.replace("-support-triage-lora", "-support-triage-GGUF")
        gguf_local = Path(snapshot_download(gguf_repo, allow_patterns=["*.gguf"]))
        target = ROOT / "models" / system
        target.mkdir(parents=True, exist_ok=True)
        for f in gguf_local.rglob("*.gguf"):
            shutil.copy(f, target / f.name)
            print("GGUF:", (target / f.name).relative_to(ROOT), f"{f.stat().st_size / 1e6:.0f} MB")
        print(f"next: bash scripts/ollama_create.sh {system}")


if __name__ == "__main__":
    main()
