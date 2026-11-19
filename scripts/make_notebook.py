"""Writes notebooks/finetune_qwen3_unsloth.ipynb (kept as code so the notebook is easy to review).

    python scripts/make_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "finetune_qwen3_unsloth.ipynb"
cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


md("""
# Fine-tune Qwen3 with LoRA (Unsloth) for support-message triage — then evaluate, quantise, publish

<a href="https://colab.research.google.com/github/lazareaya/llm-finetune-vs-api/blob/main/notebooks/finetune_qwen3_unsloth.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Adapted from the [Unsloth notebooks](https://github.com/unslothai/notebooks) (`Qwen3_(4B)-Instruct`, `Qwen3_(1.7B)-Classification`).
The task, data and evaluation come from this repository: a customer message in, a JSON record out
(`category`, `intent`, `order_number`), trained on the public Bitext dataset.

**Before "Run all"**
1. *Runtime → Change runtime type → T4 GPU*.
2. Add your Hugging Face token (write access, https://huggingface.co/settings/tokens) as a Colab secret
   named `HF_TOKEN` (key icon in the left bar, toggle *Notebook access* on). Without it the notebook asks for it.
3. Pick the model size in the next cell (`0.6B` first, `1.7B` if you have time).

**What it does** (≈ 25 min for 0.6B on a T4, ≈ 45 min for 1.7B):
data → LoRA training (rank 16, 1 epoch, 8,000 examples) → adapter pushed to your Hugging Face account →
evaluation with the repo's own scorer (1,000 test + 60 hand-written messages) → latency on the T4 →
4-bit GGUF export (q4_k_m) pushed to Hugging Face → results uploaded next to the model and offered as a zip.
""")

code("""
MODEL_SIZE = "0.6B"          # "0.6B" or "1.7B"
GITHUB_REPO = "https://github.com/lazareaya/llm-finetune-vs-api.git"
EXPORT_GGUF = True           # 4-bit GGUF for Ollama (adds ~5-10 min: llama.cpp is compiled once)
RUN_BASE_MODEL_EVAL = True   # also score the untouched Qwen3 with instructions only (for the table)

SYSTEM = f"qwen3-{MODEL_SIZE.lower()}"                    # name used by the repo, e.g. qwen3-0.6b
HF_REPO_NAME = f"{SYSTEM}-support-triage-lora"            # adapter repo on your HF account
HF_GGUF_REPO_NAME = f"{SYSTEM}-support-triage-GGUF"       # 4-bit GGUF repo
print(SYSTEM, HF_REPO_NAME, HF_GGUF_REPO_NAME)
""")

md("## 1. Install (same cell as the official Unsloth notebooks)")
code("""
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth
    !unsloth install-kernels
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.57.6
!pip install --no-deps trl==0.22.2
!pip install pydantic scikit-learn httpx openai
""")

md("## 2. Get the repository and build the same data splits (seeded, identical to local runs)")
code("""
import os, sys
os.chdir("/content")
if not os.path.exists("llm-finetune-vs-api"):
    !git clone -q {GITHUB_REPO}
os.chdir("/content/llm-finetune-vs-api")
sys.path.insert(0, os.getcwd())
!python -m triage.cli data
!md5sum data/test.jsonl
""")

md("## 3. Hugging Face login")
code("""
from huggingface_hub import login, whoami
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass("Hugging Face token (write access): ")
login(HF_TOKEN)
HF_USER = whoami()["name"]
os.environ["HF_TOKEN"] = HF_TOKEN
print("logged in as", HF_USER)
""")

md("## 4. Load Qwen3 with Unsloth and add LoRA (rank 16, same target layers as the original 135M run)")
code("""
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/Qwen3-{MODEL_SIZE}",
    max_seq_length = 256,      # a message + its record fit easily
    load_in_4bit = False,      # small models: plain 16-bit LoRA, no QLoRA needed on a T4
    load_in_8bit = False,
    full_finetuning = False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 7,
)
model.print_trainable_parameters()
print("eos:", tokenizer.eos_token, tokenizer.eos_token_id)
""")

md("""
## 5. Data: the repo's own encoding
`triage.train.encode_example` builds `prompt + record + <eos>` and masks the prompt out of the loss,
exactly as in the original CPU training — only the JSON answer is learned.
""")
code("""
from datasets import Dataset
from triage.data import read_jsonl
from triage.train import encode_example

tokenizer.padding_side = "right"
def encode(rows):
    out = [encode_example(tokenizer, r["message"], r["target"], 256) for r in rows]
    return Dataset.from_list([{**e, "attention_mask": [1] * len(e["input_ids"])} for e in out])

train_ds = encode(read_jsonl("data/train.jsonl"))
val_ds = encode(read_jsonl("data/val.jsonl"))
print(train_ds, val_ds)
print(tokenizer.decode(train_ds[0]["input_ids"]))
""")

md("## 6. Train (1 epoch = 500 steps of 16 examples)")
code("""
import time
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    args = SFTConfig(
        output_dir = "runs",
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        per_device_eval_batch_size = 32,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        warmup_steps = 30,
        lr_scheduler_type = "cosine",
        logging_steps = 25,
        eval_strategy = "steps",
        eval_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        seed = 7,
        report_to = "none",
        remove_unused_columns = False,
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 256,
    ),
)
started = time.time()
stats = trainer.train()
TRAIN_MINUTES = round((time.time() - started) / 60, 1)
print("training minutes:", TRAIN_MINUTES)
""")

md("## 7. Save the adapter and push it to Hugging Face")
code("""
import json
ADAPTER_DIR = f"adapters/{SYSTEM}"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log = {
    "base_model": f"Qwen/Qwen3-{MODEL_SIZE} (loaded as unsloth/Qwen3-{MODEL_SIZE})",
    "framework": "Unsloth + TRL SFTTrainer, LoRA",
    "config": {"rank": 16, "alpha": 32, "dropout": 0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
               "epochs": 1, "batch_size": 16, "learning_rate": 2e-4, "warmup_steps": 30, "scheduler": "cosine",
               "max_length": 256, "seed": 7},
    "train_examples": len(train_ds), "val_examples": len(val_ds),
    "steps": stats.global_step, "final_train_loss": round(stats.training_loss, 4),
    "trainable_parameters": trainable, "total_parameters": total,
    "trainable_pct": round(100 * trainable / total, 2),
    "training_minutes": TRAIN_MINUTES,
    "gpu": torch.cuda.get_device_name(0),
    "history": trainer.state.log_history,
}
with open(f"{ADAPTER_DIR}/training_log.json", "w") as f:
    json.dump(log, f, indent=2)
model.push_to_hub(f"{HF_USER}/{HF_REPO_NAME}", token=HF_TOKEN)
tokenizer.push_to_hub(f"{HF_USER}/{HF_REPO_NAME}", token=HF_TOKEN)
print(f"https://huggingface.co/{HF_USER}/{HF_REPO_NAME}")
""")

md("""
## 8. Evaluate with the repo's scorer (separate process, plain Transformers + PEFT, merged adapter)
Same messages, same checks and same metrics as every other system in the comparison.
""")
code("""
env_var = "TRIAGE_ADAPTER_" + SYSTEM.upper().replace("-", "_").replace(".", "_")
os.environ[env_var] = os.path.abspath(ADAPTER_DIR)
torch.cuda.empty_cache()
!python -m evals.run --systems {SYSTEM}-lora --set test
!python -m evals.run --systems {SYSTEM}-lora --set challenge
!python -m evals.latency --system {SYSTEM}-lora
if RUN_BASE_MODEL_EVAL:
    !python -m evals.run --systems {SYSTEM}-prompt --set test --limit 100
    !python -m evals.run --systems {SYSTEM}-prompt --set challenge
""")

md("## 9. Upload the results next to the model (and keep a zip)")
code("""
import json, shutil
from huggingface_hub import HfApi
report = json.load(open("evals/report.json"))
mine = {k: v for k, v in report.items() if SYSTEM in k}
os.makedirs("colab_results/outputs", exist_ok=True)
json.dump(mine, open("colab_results/report_entries.json", "w"), indent=2)
for name in os.listdir("evals/outputs"):
    if SYSTEM in name:
        shutil.copy(f"evals/outputs/{name}", "colab_results/outputs/")
shutil.copy(f"{ADAPTER_DIR}/training_log.json", "colab_results/")
api = HfApi(token=HF_TOKEN)
api.upload_folder(folder_path="colab_results", path_in_repo="eval", repo_id=f"{HF_USER}/{HF_REPO_NAME}")
for k, v in mine.items():
    if "exact_record_pct" in v:
        print(f"{k:28s} exact={v['exact_record_pct']}%  valid={v['valid_record_pct']}%  n={v['n']}")
    else:
        print(k, v)
""")

md("## 10. 4-bit quantisation (GGUF q4_k_m) for Ollama / llama.cpp, pushed to Hugging Face")
code("""
if EXPORT_GGUF:
    model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method = "q4_k_m")
    model.push_to_hub_gguf(f"{HF_USER}/{HF_GGUF_REPO_NAME}", tokenizer,
                           quantization_method = "q4_k_m", token = HF_TOKEN)
    !ls -la gguf_model* 2>/dev/null; find / -name "*.gguf" -newer data/test.jsonl 2>/dev/null | head
    print(f"https://huggingface.co/{HF_USER}/{HF_GGUF_REPO_NAME}")
""")

md("## 11. Download a zip with the adapter and results (backup)")
code("""
shutil.make_archive(f"/content/{SYSTEM}_results", "zip", ".", "colab_results")
shutil.make_archive(f"/content/{SYSTEM}_adapter", "zip", ".", ADAPTER_DIR)
try:
    from google.colab import files
    files.download(f"/content/{SYSTEM}_results.zip")
    files.download(f"/content/{SYSTEM}_adapter.zip")
except Exception as e:
    print("download from the file browser:", e)
print("Done. Back on the Mac:  python scripts/fetch_colab_results.py", f"{HF_USER}/{HF_REPO_NAME}")
""")


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in cells
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
