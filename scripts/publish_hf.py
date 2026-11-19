"""Publish the model card and the technical report on the Hugging Face model page.

    hf auth login                      # once, with a write token
    python scripts/publish_hf.py <hf_user>/qwen3-0.6b-support-triage-lora

Uploads README.md (model card: what it does, how to use it, results table, limits, credits)
plus report/report.pdf and report/report.md to the adapter repo, and a short card to the GGUF repo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GITHUB = "https://github.com/lazareaya/llm-finetune-vs-api"


def results_block() -> str:
    from evals.final_table import build_tables

    return build_tables(json.loads((ROOT / "evals" / "report.json").read_text()))


def card(repo_id: str, base: str, gguf_repo: str) -> str:
    return f"""---
base_model: {base}
library_name: peft
license: apache-2.0
language: [en]
datasets: [bitext/Bitext-customer-support-llm-chatbot-training-dataset]
pipeline_tag: text-generation
tags: [lora, peft, unsloth, trl, json, structured-output, information-extraction, customer-support, llm-as-a-judge]
---

# {repo_id.split('/')[1]}

LoRA adapter for **{base}** that reads a customer-support message and writes a JSON routing record:

```json
{{"category": "REFUND", "intent": "get_refund", "order_number": "PX300127"}}
```

11 categories, 27 intents, order number copied exactly from the message (or `null`). Trained with
**Unsloth + TRL** on a free Colab T4 (rank 16, 1 epoch, 8,000 examples from the Bitext dataset).
It is compared with **Gemini Flash** (API), the untouched model, a 135M fine-tuned model and a classic classifier,
with exact metrics, a blind **LLM judge**, latency and **cost per 1,000 requests**.

- Code, evaluation and raw outputs: {GITHUB}
- Technical report: [report.pdf](report.pdf) ([Markdown](report.md))
- 4-bit GGUF (q4_k_m) for Ollama / llama.cpp: https://huggingface.co/{gguf_repo}

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "{repo_id}").merge_and_unload()
prompt = "### Message\\nThe kettle arrived broken. I want my money back for order PX300127.\\n### Record\\n"
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
out = model.generate(**ids, max_new_tokens=48, do_sample=False)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use the plain prompt above, not the chat template. In production, validate the output: the JSON must parse,
the labels must be known, and the order number must appear in the message. Otherwise, send the message to a
person. `triage/schema.py` in the repository does this.

## Results

{results_block()}

## Limitations

The training data is template-generated, so held-out accuracy is optimistic; the 60-message hand-written
set is the more realistic number. There is no "other" label, so off-topic messages get the nearest intent.
English only, one order number per message, one training seed. See the report for details.

## Credits

Built on [BelAnas/small-llm-lora-finetune]({"https://github.com/BelAnas/small-llm-lora-finetune"}) (MIT) and
[Unsloth](https://github.com/unslothai/unsloth). Data: Bitext customer-support dataset (CDLA-Sharing-1.0).
"""


def main() -> None:
    repo_id = sys.argv[1]
    system = repo_id.split("/")[1].removesuffix("-support-triage-lora")
    from triage.config import QWEN_MODELS

    base = QWEN_MODELS[system][0]
    gguf_repo = repo_id.replace("-support-triage-lora", "-support-triage-GGUF")
    api = HfApi()
    api.upload_file(path_or_fileobj=card(repo_id, base, gguf_repo).encode(), path_in_repo="README.md", repo_id=repo_id)
    for name in ("report.pdf", "report.md"):
        path = ROOT / "report" / name
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id)
    gguf_card = f"""---
base_model: {repo_id}
license: apache-2.0
tags: [gguf, ollama, llama.cpp, quantized, q4_k_m]
---

# {gguf_repo.split('/')[1]}

{base} fine-tuned with LoRA for customer-support triage ([adapter and full card](https://huggingface.co/{repo_id})),
merged and quantised to 4 bits (GGUF q4_k_m) with Unsloth.

```bash
# Modelfile:  FROM ./<file>.gguf / TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\" / PARAMETER temperature 0 / PARAMETER stop "<|im_end|>"
ollama create triage -f Modelfile
curl http://127.0.0.1:11434/api/generate -d '{{"model":"triage","raw":true,"stream":false,"prompt":"### Message\\ncancel order ORD-12345\\n### Record\\n"}}'
```

Latency and accuracy of this file are in {GITHUB}.
"""
    try:
        api.upload_file(path_or_fileobj=gguf_card.encode(), path_in_repo="README.md", repo_id=gguf_repo)
    except Exception as exc:  # the GGUF repo is optional
        print("GGUF card not uploaded:", exc)
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
