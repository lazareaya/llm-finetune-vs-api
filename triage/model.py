"""Loading the model and generating text, on a CPU or on a GPU (CUDA or Apple MPS)."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """TRIAGE_DEVICE=cpu|cuda|mps forces a device; by default the fastest one available is used."""
    forced = os.getenv("TRIAGE_DEVICE", "").lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_threads(threads: int) -> None:
    if threads > 0:
        torch.set_num_threads(threads)


def load_tokenizer(name_or_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps the end of every prompt next to the first generated token,
    # which batched generation with a decoder-only model requires.
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    base_model: str | Path, adapter_dir: str | Path | None = None, merge: bool = True, device: str | None = None
):
    """Load the base model, and the LoRA add-on if a folder is given.

    With merge=True the add-on is folded into the base weights, so serving costs exactly
    the same as the untouched model: no extra layers at inference time.
    """
    device = device or "cpu"
    # Half precision on a GPU (bfloat16 keeps the range of float32); full precision on a CPU.
    if device == "cuda":
        # bfloat16 needs an Ampere GPU or newer; a Colab T4 uses float16 instead.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(base_model), dtype=dtype).to(device)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model


def has_adapter(adapter_dir: str | Path) -> bool:
    """A local folder with an adapter, or a Hugging Face Hub repo id such as user/name."""
    if (Path(adapter_dir) / "adapter_config.json").exists():
        return True
    text = str(adapter_dir)
    return not Path(text).exists() and text.count("/") == 1 and not text.startswith((".", "/"))


@torch.inference_mode()
def generate(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 48, batch_size: int = 16
) -> list[str]:
    """Greedy generation in batches. Greedy, because routing must give the same answer twice."""
    outputs: list[str] = []
    # Sorting by length puts similar prompts in one batch, so little time is spent on padding.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: dict[int, str] = {}
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer(
            [prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        for i, text in zip(idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            results[i] = text.strip()
    outputs = [results[i] for i in range(len(prompts))]
    return outputs


def render_chat(tokenizer, turns: list[dict]) -> str:
    """Turn chat messages into the prompt text the base model was trained to expect."""
    # enable_thinking=False turns off the hidden reasoning of Qwen3; other templates ignore it.
    return tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=False)
