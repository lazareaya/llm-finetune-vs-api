"""The final comparison table: accuracy, valid JSON, judge grade, latency and cost per 1,000 requests.

    python -m evals.final_table            # print the tables
    python -m evals.final_table --write    # also refresh them in README.md and report/report.md

Prices are list prices for the hosted model (per million tokens), checked on the date below.
Local systems cost nothing per request in API fees; their cost is the machine they run on,
estimated here from the measured latency and an hourly machine price.
"""
from __future__ import annotations

import argparse
import json
import re

from evals.run import REPORT
from triage.config import ROOT

PRICES_CHECKED = "2026-11-19"
PRICES_SOURCE = "https://aistudio.google.com/docs/pricing"
# USD per 1M tokens (standard paid tier). Output price includes thinking tokens.
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),  # doubles to 1.50 / 7.50 on 2027-01-01
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3-flash-preview": (0.50, 3.00),  # legacy Flash; used here because of free-tier daily quotas
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}
# Machine prices for the local systems' compute estimate (USD per hour).
MACHINE_PER_HOUR = {"cpu": 0.05, "mps": 0.05, "ollama": 0.05, "cuda": 0.35}
MACHINE_NOTE = (
    "Local compute is estimated at $0.05/h for a small CPU machine or laptop share and $0.35/h "
    "for a cloud T4 GPU, times the measured time per message (one machine, no parallelism)."
)

ROWS = [
    ("base-prompt", "SmolLM2-135M, instructions only", "135M"),
    ("base-examples", "SmolLM2-135M, instructions + 8 examples", "135M"),
    ("lora", "SmolLM2-135M fine-tuned (LoRA, original repo)", "135M"),
    ("qwen3-0.6b-prompt", "Qwen3-0.6B, instructions only", "0.6B"),
    ("qwen3-0.6b-lora", "**Qwen3-0.6B fine-tuned (LoRA, Unsloth)**", "0.6B"),
    ("qwen3-1.7b-lora", "Qwen3-1.7B fine-tuned (LoRA, Unsloth)", "1.7B"),
    ("ollama-triage-smollm2-135m", "SmolLM2-135M fine-tuned, 4-bit GGUF in Ollama", "135M (Q4_K_M)"),
    ("ollama-triage-qwen3-0.6b", "**Qwen3-0.6B fine-tuned, 4-bit GGUF in Ollama (Mac)**", "0.6B (Q4_K_M, 396 MB)"),
    ("ollama-triage-qwen3-1.7b", "Qwen3-1.7B fine-tuned, 4-bit GGUF in Ollama", "1.7B (Q4_K_M)"),
    ("classic", "TF-IDF + logistic regression + regex", "-"),
    ("hosted", "Gemini 3 Flash (API), instructions only", "large (undisclosed)"),
]

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def latency_cell(report: dict, system: str, r: dict) -> tuple[str, float | None, str]:
    """Best available latency: one request at a time if measured, else batched time per message."""
    single = report.get(f"latency/{system}")
    if single:
        cell = f"{single['median_ms']:.0f} ms ({single['where']}, 1 request)"
        cpu = report.get(f"latency/{system}-cpu")
        if cpu:
            cell += f"; {cpu['median_ms']:.0f} ms CPU only"
        return cell, single["median_ms"], single.get("device", "cpu")
    if r.get("median_ms_per_request") is not None:
        where = "API" if system == "hosted" else "Mac CPU, Ollama"
        dev = "api" if system == "hosted" else "ollama"
        return f"{r['median_ms_per_request']:.0f} ms ({where}, 1 request)", r["median_ms_per_request"], dev
    if system == "lora" and "latency" in report:  # the original repo's measurement
        ms = report["latency"]["merged"]["median_ms"]
        return f"{ms:.0f} ms (CPU, 1 request)", ms, "cpu"
    if r.get("ms_per_message_batched") is not None:
        dev = r.get("device", "cpu")
        label = {"cuda": "T4 GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(dev, dev)
        if system in ("classic", "mock"):
            return f"{r['ms_per_message_batched']:.2f} ms (CPU)", r["ms_per_message_batched"], "cpu"
        return f"{r['ms_per_message_batched']:.0f} ms ({label}, batched)", r["ms_per_message_batched"], dev
    return "-", None, "cpu"


def cost_cell(system: str, r: dict, ms: float | None, device: str) -> str:
    if system == "hosted":
        model = r.get("model", "")
        price = PRICES.get(model)
        if not price or not r.get("n"):
            return "see note"
        single = REPORT_CACHE.get("latency/hosted")
        if single and single.get("model") == model:  # measured one message per request
            per_req_in, per_req_out = single["input_tokens_per_request"], single["output_tokens_per_request"]
        else:
            per_req_in, per_req_out = r["input_tokens"] / r["n"], r["output_tokens"] / r["n"]
        usd = 1000 * (per_req_in * price[0] + per_req_out * price[1]) / 1e6
        return f"${usd:.2f} API ({per_req_in:.0f} in + {per_req_out:.0f} out tokens/req)"
    if ms is None:
        return "$0 API"
    compute = 1000 * ms / 1000 / 3600 * MACHINE_PER_HOUR.get(device, 0.05)
    return f"$0 API (+ ~${compute:.4f} compute)"


REPORT_CACHE: dict = {}


def build_tables(report: dict) -> str:
    REPORT_CACHE.clear()
    REPORT_CACHE.update(report)
    judge = report.get("judge/test", {}).get("per_system", {})
    out = []
    out.append("**Held-out test set** (public dataset messages never used for training)\n")
    out.append("| System | Size | Whole record correct | Valid JSON | Judge grade (1-5) | Latency per message | Cost per 1,000 requests |")
    out.append("|---|---|---|---|---|---|---|")
    for system, label, size in ROWS:
        r = report.get(f"{system}/test")
        if not r:
            continue
        lat, ms, dev = latency_cell(report, system, r)
        j = judge.get(system)
        jcell = f"{j['mean_grade']:.2f} (n={j['n']})" if j else "-"
        out.append(
            f"| {label} | {size} | {r['exact_record_pct']}% (n={r['n']}) | {r['valid_record_pct']}% | {jcell} | "
            f"{lat} | {cost_cell(system, r, ms, dev)} |"
        )
    out.append("")
    out.append("**Hand-written set** (60 messages in a different style, `evals/challenge.jsonl`)\n")
    out.append("| System | Whole record correct | Intent correct | Order number found | Valid JSON |")
    out.append("|---|---|---|---|---|")
    for system, label, _ in ROWS:
        r = report.get(f"{system}/challenge")
        if r:
            pct = lambda v: "-" if v is None else f"{v}%"  # noqa: E731
            out.append(f"| {label} | {pct(r['exact_record_pct'])} | {pct(r['intent_accuracy_pct'])} | "
                       f"{pct(r['order_number_found_pct'])} | {pct(r['valid_record_pct'])} |")
    agreement = report.get("judge/test", {}).get("agreement")
    if agreement:
        jt = report["judge/test"]
        out.append("")
        out.append(f"**Judge vs exact metrics** ({jt['judge_model']}, {jt['messages']} test messages, "
                   f"{jt['judged_outputs']} graded outputs): grade 4-5 agrees with an exactly correct record "
                   f"{agreement['binary_agreement_pct']}% of the time (Cohen's kappa {agreement['cohen_kappa']}); "
                   f"mean grade {agreement['mean_grade_when_exact']} when exact, {agreement['mean_grade_when_wrong']} "
                   f"when wrong; {agreement['exact_but_graded_1_to_3']} exact records graded 1-3, "
                   f"{agreement['wrong_but_graded_4_or_5']} wrong records graded 4-5.")
    hosted = report.get("hosted/test")
    model = hosted.get("model") if hosted else None
    price = f"{model}: ${PRICES[model][0]} input / ${PRICES[model][1]} output per 1M tokens" if model in PRICES else "Gemini Flash list price"
    out.append("")
    out.append(f"Cost: {price} ({PRICES_SOURCE}, checked {PRICES_CHECKED}; thinking tokens are billed as output). {MACHINE_NOTE}")
    return "\n".join(out)


def write_into(path, tables: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    if START not in text:
        return False
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{tables}\n{END}", text, flags=re.DOTALL)
    path.write_text(new)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh the tables in README.md and report/report.md")
    args = parser.parse_args()
    report = json.loads(REPORT.read_text())
    tables = build_tables(report)
    print(tables)
    if args.write:
        for path in (ROOT / "README.md", ROOT / "report" / "report.md"):
            print(f"{'updated' if write_into(path, tables) else 'no markers in'} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
