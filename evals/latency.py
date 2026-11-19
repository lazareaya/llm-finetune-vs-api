"""Time the fine-tuned model one message at a time, the way the API calls it.

    python -m evals.latency                           # original 135M model: merged vs unmerged, CPU
    python -m evals.latency --system qwen3-0.6b-lora  # fine-tuned Qwen3 (GPU if available)
    python -m evals.latency --system ollama           # 4-bit GGUF served by Ollama on this machine

Results go into evals/report.json under "latency" (original) or "latency/<system>".
"""
from __future__ import annotations

import json
import statistics
import time

import torch

from evals.run import REPORT, SETS
from triage.config import get_settings
from triage.data import read_jsonl
from triage.model import generate, load_model, load_tokenizer, set_threads
from triage.prompts import ft_prompt


def time_model(model, tokenizer, prompts: list[str], max_new_tokens: int) -> dict:
    generate(model, tokenizer, prompts[:2], max_new_tokens, batch_size=1)  # warm-up, not timed
    times = []
    for prompt in prompts:
        started = time.perf_counter()
        generate(model, tokenizer, [prompt], max_new_tokens, batch_size=1)
        times.append(1000 * (time.perf_counter() - started))
    times.sort()
    return {
        "messages": len(times),
        "median_ms": round(statistics.median(times)),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))]),
    }


def time_ollama(messages: list[str]) -> dict:
    from triage.providers import OllamaProvider

    settings = get_settings()
    provider = OllamaProvider(settings.ollama_model, settings.ollama_url, settings.max_new_tokens)
    provider.complete(messages[:2])  # warm-up: loads the model into memory, not timed
    provider.latencies_ms.clear()
    provider.complete(messages)
    times = sorted(provider.latencies_ms)
    return {
        "messages": len(times),
        "median_ms": round(statistics.median(times)),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))]),
        "model": settings.ollama_model,
    }


def main() -> None:
    import argparse
    import platform

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="lora", help="lora (original 135M), qwen3-0.6b-lora, qwen3-1.7b-lora or ollama")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    settings = get_settings()
    messages = [ex["message"] for ex in read_jsonl(SETS["challenge"])[: args.n]]
    machine = f"{platform.system()} {platform.machine()}"
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}

    if args.system == "ollama":
        result = time_ollama(messages)
        result.update(device="ollama", where=f"{machine}, CPU+GPU via Ollama, 4-bit")
        print("ollama", result)
        report[f"latency/ollama-{result['model']}"] = result
    elif args.system == "lora":  # the original measurement: merged vs unmerged adapter on the CPU
        set_threads(settings.threads)
        tokenizer = load_tokenizer(settings.base_model)
        prompts = [ft_prompt(m) for m in messages]
        result = {"cpu_threads": torch.get_num_threads()}
        for name, merge in [("merged", True), ("unmerged", False)]:
            model = load_model(settings.base_model, settings.adapter_dir, merge=merge)
            result[name] = time_model(model, tokenizer, prompts, settings.max_new_tokens)
            print(name, result[name])
        report["latency"] = result
    else:
        from triage.config import qwen_model
        from triage.model import pick_device

        base, adapter = qwen_model(args.system.removesuffix("-lora"))
        device = pick_device()
        tokenizer = load_tokenizer(base)
        model = load_model(base, adapter, merge=True, device=device)
        result = time_model(model, tokenizer, [ft_prompt(m) for m in messages], settings.max_new_tokens)
        label = {"cuda": "T4 GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(device, device)
        result.update(device=device, where=f"{label}, 16-bit")
        print(args.system, result)
        report[f"latency/{args.system}"] = result
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
