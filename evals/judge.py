"""LLM-as-a-judge: a large model grades every system's record from 1 to 5, then we check
whether its grades agree with the exact metrics.

    export OPENAI_API_KEY=<Google AI Studio key>
    export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    python -m evals.judge --set test --limit 100

How it works
- It reads the raw outputs saved by evals/run.py in evals/outputs/<system>__<set>.jsonl and keeps
  the first N messages that every selected system has answered.
- One request per message: the judge sees the message, the reference record and every system's
  output at once, under shuffled anonymous letters (A, B, C...). It does not know which system
  wrote what, and the order changes per message, so it cannot favour a system by name or position.
- It returns a 1-5 grade per letter. Answers are cached in evals/cache/judge.jsonl, so a run that
  hits a free-tier rate limit can simply be started again.

Agreement with the exact metrics is written to evals/judge_report.json:
- mean grade per system, next to its exact-record accuracy;
- how often a grade of 4-5 coincides with an exactly correct record (and 1-3 with a wrong one),
  and Cohen's kappa for that binary agreement;
- the disagreements (exact but graded low, wrong but graded high), with examples.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

from evals.run import OUTPUTS, REPORT
from triage.config import ROOT
from triage.schema import check_output

CACHE = ROOT / "evals" / "cache" / "judge.jsonl"
JUDGE_REPORT = ROOT / "evals" / "judge_report.json"
DEFAULT_SYSTEMS = [
    "base-prompt", "base-examples", "lora", "qwen3-0.6b-prompt", "qwen3-0.6b-lora",
    "qwen3-1.7b-lora", "classic", "hosted", "ollama-triage-qwen3-0.6b",
]

RUBRIC = """You grade the output of customer-support routing systems.
Each system read a customer message and had to answer with ONE JSON record:
{"category": ..., "intent": ..., "order_number": ...}
order_number must be the order number written in the message, copied exactly, or null.

You get the message, the reference record (the correct answer) and several candidate outputs
labelled with letters. Grade every candidate from 1 to 5:
5 = same meaning as the reference on all three fields (category, intent, order number).
4 = usable: one field slightly off but harmless (e.g. a very close intent in the right category),
    or the intent is arguably as correct as the reference for this message.
3 = partly right: right category but a clearly different intent, or a wrong/missing order number.
2 = mostly wrong: wrong category and intent, but a JSON record was produced.
1 = unusable: no valid JSON record, invented labels, or text that is not a record.
Judge the content, not the formatting (spaces, key order). Be strict and consistent.
Answer with JSON only: {"grades": {"A": <1-5>, "B": <1-5>, ...}, "notes": "<one short sentence>"}"""


def load_outputs(systems: list[str], set_name: str) -> dict[str, list[dict]]:
    found = {}
    for system in systems:
        path = OUTPUTS / f"{system}__{set_name}.jsonl"
        if path.exists():
            found[system] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            print(f"skip {system}: no outputs in {path.name} (run evals.run for it first)")
    return found


def _cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    return {row["key"]: row for row in rows}


def ask_judge(client, model: str, message: str, reference: str, candidates: dict[str, str],
              reasoning_effort: str, retries: int = 10) -> dict:
    listing = "\n".join(f"{letter}: {text[:300]!s}" for letter, text in candidates.items())
    user = f"Message:\n{message}\n\nReference record:\n{reference}\n\nCandidates:\n{listing}"
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "2048")),
        response_format={"type": "json_object"},
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            grades = {k: int(v) for k, v in parsed.get("grades", {}).items() if str(v).strip().isdigit()}
            if set(grades) >= set(candidates):
                usage = getattr(response, "usage", None)
                return {
                    "grades": grades,
                    "notes": parsed.get("notes", ""),
                    "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output_tokens": (getattr(usage, "total_tokens", 0) - getattr(usage, "prompt_tokens", 0)) if usage else 0,
                }
            if attempt == retries:
                raise ValueError(f"judge gave no grade for every candidate: {text[:200]}")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if attempt == retries or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError("unreachable")


def ask_judge_group(client, model: str, items: list[tuple[str, str, dict[str, str]]],
                    reasoning_effort: str, retries: int = 10) -> list[dict]:
    """Several messages in one request, for free tiers with a small daily request quota.
    Each message keeps its own reference and its own shuffled letters."""
    if len(items) == 1:
        return [ask_judge(client, model, *items[0], reasoning_effort, retries)]
    blocks = []
    for k, (message, reference, candidates) in enumerate(items, 1):
        listing = "\n".join(f"{letter}: {text[:300]!s}" for letter, text in candidates.items())
        blocks.append(f"=== Item {k} ===\nMessage:\n{message}\n\nReference record:\n{reference}\n\nCandidates:\n{listing}")
    rubric = RUBRIC.replace(
        'Answer with JSON only: {"grades": {"A": <1-5>, "B": <1-5>, ...}, "notes": "<one short sentence>"}',
        'There are several independent items; grade each one separately. Answer with JSON only: '
        '{"items": {"1": {"grades": {"A": <1-5>, ...}, "notes": "<one short sentence>"}, "2": {...}, ...}}')
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": rubric}, {"role": "user", "content": "\n\n".join(blocks)}],
        temperature=0,
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "16384")),
        response_format={"type": "json_object"},
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            parsed = (json.loads(match.group(0)) if match else {}).get("items", {})
            rows = []
            for k, (_, _, candidates) in enumerate(items, 1):
                entry = parsed.get(str(k), {})
                grades = {a: int(v) for a, v in entry.get("grades", {}).items() if str(v).strip().isdigit()}
                if not set(grades) >= set(candidates):
                    raise ValueError(f"item {k}: no grade for every candidate")
                rows.append({"grades": grades, "notes": entry.get("notes", "")})
            usage = getattr(response, "usage", None)
            tin = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            tout = ((getattr(usage, "total_tokens", 0) or 0) - tin) if usage else 0
            for row in rows:
                row["input_tokens"], row["output_tokens"] = tin / len(rows), tout / len(rows)
            return rows
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if attempt == retries or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError("unreachable")


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else None


def is_exact(row: dict) -> bool:
    expected = json.loads(row["expected"])
    record = check_output(row["output"], row["message"]).record
    return bool(record) and (record.category, record.intent, record.order_number) == (
        expected["category"], expected["intent"], expected["order_number"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="set_name", default="test", choices=["test", "challenge"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gemini-3.8-flash")
    parser.add_argument("--reasoning-effort", default=os.getenv("JUDGE_REASONING_EFFORT", "low"))
    parser.add_argument("--per-call", type=int, default=int(os.getenv("JUDGE_PER_CALL", "1")),
                        help="messages graded per request (free tiers allow few requests per day)")
    args = parser.parse_args()

    from openai import OpenAI

    outputs = load_outputs(args.systems, args.set_name)
    systems = list(outputs)
    n = min([args.limit] + [len(rows) for rows in outputs.values()])
    for system, rows in outputs.items():  # every system must have answered the same messages
        assert [r["message"] for r in rows[:n]] == [r["message"] for r in outputs[systems[0]][:n]], system

    client, cache = OpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT", "120")), max_retries=0), _cache()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    grades: dict[str, list[int]] = {s: [] for s in systems}
    exact: dict[str, list[bool]] = {s: [] for s in systems}
    disagreements: list[dict] = []
    judge_in = judge_out = 0

    plan = []
    for i in range(n):
        first = outputs[systems[0]][i]
        order = systems[:]
        random.Random(f"{args.set_name}-{i}").shuffle(order)  # blind and order-shuffled per message
        letters = {chr(65 + k): s for k, s in enumerate(order)}
        candidates = {letter: outputs[s][i]["output"] for letter, s in letters.items()}
        key = json.dumps([args.model, args.reasoning_effort, first["message"], candidates], ensure_ascii=False)
        plan.append((i, first, letters, candidates, key))

    todo = [p for p in plan if p[4] not in cache]
    for start in range(0, len(todo), args.per_call):
        chunk = todo[start : start + args.per_call]
        rows = ask_judge_group(client, args.model, [(p[1]["message"], p[1]["expected"], p[3]) for p in chunk],
                               args.reasoning_effort)
        with CACHE.open("a") as fh:
            for p, row in zip(chunk, rows):
                row = {"key": p[4], **row}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache[p[4]] = row
        print(f"\r{start + len(chunk)}/{len(todo)} new messages graded", end="", flush=True)
    print()

    for i, first, letters, candidates, key in plan:
        row = cache[key]
        judge_in += row.get("input_tokens", 0)
        judge_out += row.get("output_tokens", 0)
        for letter, system in letters.items():
            grade = row["grades"][letter]
            ok = is_exact(outputs[system][i])
            grades[system].append(grade)
            exact[system].append(ok)
            if ok != (grade >= 4):
                disagreements.append({
                    "system": system, "message": first["message"], "expected": first["expected"],
                    "output": outputs[system][i]["output"][:200], "grade": grade, "exact": ok,
                    "judge_notes": row.get("notes", ""),
                })

    all_grades = [g for s in systems for g in grades[s]]
    all_exact = [e for s in systems for e in exact[s]]
    high = [g >= 4 for g in all_grades]
    per_system = {}
    for s in systems:
        g, e = grades[s], exact[s]
        per_system[s] = {
            "n": len(g),
            "mean_grade": round(statistics.mean(g), 2),
            "grade_4_or_5_pct": round(100 * sum(x >= 4 for x in g) / len(g), 1),
            "exact_record_pct": round(100 * sum(e) / len(e), 1),
            "grade_distribution": {str(k): g.count(k) for k in range(1, 6)},
        }
    exact_grades = [g for g, e in zip(all_grades, all_exact) if e]
    wrong_grades = [g for g, e in zip(all_grades, all_exact) if not e]
    result = {
        "judge_model": args.model,
        "messages_per_request": args.per_call,
        "reasoning_effort": args.reasoning_effort or None,
        "set": args.set_name,
        "messages": n,
        "judged_outputs": len(all_grades),
        "judge_input_tokens": judge_in,
        "judge_output_tokens": judge_out,
        "per_system": per_system,
        "agreement": {
            "binary_agreement_pct": round(100 * sum(h == e for h, e in zip(high, all_exact)) / len(high), 1),
            "cohen_kappa": cohen_kappa(high, all_exact),
            "mean_grade_when_exact": round(statistics.mean(exact_grades), 2) if exact_grades else None,
            "mean_grade_when_wrong": round(statistics.mean(wrong_grades), 2) if wrong_grades else None,
            "exact_but_graded_1_to_3": sum(1 for d in disagreements if d["exact"]),
            "wrong_but_graded_4_or_5": sum(1 for d in disagreements if not d["exact"]),
        },
        "disagreement_examples": disagreements[:25],
    }
    JUDGE_REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[f"judge/{args.set_name}"] = {k: v for k, v in result.items() if k != "disagreement_examples"}
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"judge: {args.model}, {n} messages, {len(all_grades)} outputs graded")
    for s, r in per_system.items():
        print(f"  {s:20s} mean grade {r['mean_grade']:.2f}   grade>=4 {r['grade_4_or_5_pct']:5.1f}%   exact {r['exact_record_pct']:5.1f}%")
    print("  agreement:", result["agreement"])


if __name__ == "__main__":
    main()
