"""Error analysis: what kind of mistake each system makes, from the saved raw outputs.

    python -m evals.errors            # print
    python -m evals.errors --write    # also refresh the block in report/report.md

Every wrong record falls in exactly one bucket, checked in this order:
no valid JSON -> unknown label -> wrong category -> right category, wrong intent ->
order number missed -> order number invented -> order number wrong.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from evals.final_table import ROWS
from evals.run import OUTPUTS
from triage.config import ROOT
from triage.labels import CATEGORIES, INTENT_TO_CATEGORY
from triage.schema import check_output

BUCKETS = [
    "no valid JSON", "unknown label", "wrong category", "right category, wrong intent",
    "order number missed", "order number invented", "order number wrong",
]
START, END = "<!-- ERRORS:START -->", "<!-- ERRORS:END -->"


def bucket(row: dict) -> str | None:
    exp = json.loads(row["expected"])
    rec = check_output(row["output"], row["message"]).record
    if rec is None:
        return "no valid JSON"
    if rec.category not in CATEGORIES or rec.intent not in INTENT_TO_CATEGORY:
        return "unknown label"
    if rec.category != exp["category"]:
        return "wrong category"
    if rec.intent != exp["intent"]:
        return "right category, wrong intent"
    if rec.order_number != exp["order_number"]:
        if exp["order_number"] and not rec.order_number:
            return "order number missed"
        if not exp["order_number"]:
            return "order number invented"
        return "order number wrong"
    return None


def analyse(set_name: str) -> tuple[list[str], dict[str, list[dict]]]:
    lines = [f"| System ({set_name} set) | n | Errors | " + " | ".join(BUCKETS) + " |",
             "|---|---|---|" + "---|" * len(BUCKETS)]
    examples: dict[str, list[dict]] = {}
    for system, label, _ in ROWS:
        path = OUTPUTS / f"{system}__{set_name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        counts: Counter[str] = Counter()
        for row in rows:
            b = bucket(row)
            if b:
                counts[b] += 1
                examples.setdefault(system, []).append({**row, "bucket": b})
        lines.append(f"| {label} | {len(rows)} | {sum(counts.values())} | "
                     + " | ".join(str(counts.get(b, 0)) for b in BUCKETS) + " |")
    return lines, examples


def build() -> str:
    out = ["Each wrong record counted once, in the first bucket that applies.", ""]
    for set_name in ("test", "challenge"):
        lines, examples = analyse(set_name)
        out += lines + [""]
    # A few concrete failures of the main systems on the hand-written set.
    _, examples = analyse("challenge")
    for system in ("qwen3-0.6b-lora", "hosted", "lora"):
        if system in examples:
            label = next(lbl for s, lbl, _ in ROWS if s == system).replace("**", "")
            out.append(f"Examples of errors, {label}, hand-written set:")
            out.append("")
            for ex in examples[system][:4]:
                msg = ex["message"].replace("|", "/")
                got = re.sub(r"\s+", " ", ex["output"])[:120].replace("|", "/")
                out.append(f"- *{msg}* → expected `{ex['expected']}`, got `{got}` ({ex['bucket']})")
            out.append("")
    return "\n".join(out).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    text = build()
    print(text)
    if args.write:
        path = ROOT / "report" / "report.md"
        doc = path.read_text()
        path.write_text(re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{text}\n{END}", doc, flags=re.DOTALL))
        print("updated", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
