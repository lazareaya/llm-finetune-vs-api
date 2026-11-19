"""Refresh the generated blocks of report/report.md and render it to report/report.pdf.

    python scripts/build_report.py

Steps: final tables (evals/final_table) and error analysis (evals/errors) are written into
README.md and report/report.md, then the Markdown is converted to HTML and printed to PDF with
a headless Chrome/Chromium (set CHROME=/path/to/chrome if it is not in the usual place).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 7.6pt; page-break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f1f3f5; }
code { font-family: Menlo, monospace; font-size: 8pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f4f4f4; padding: 6pt; font-size: 7.6pt; white-space: pre-wrap; page-break-inside: avoid; }
a { color: #1f4e9c; text-decoration: none; }
"""

CHROMES = [
    os.getenv("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or "",
]


def training_block() -> str:
    """One row per training run, from the training logs saved next to each adapter."""
    import json

    rows = []
    logs = [("SmolLM2-135M (original, CPU)", ROOT / "adapter" / "training_log.json")]
    logs += [(p.parent.name, p) for p in sorted((ROOT / "adapters").glob("*/training_log.json"))]
    for name, path in logs:
        if not path.exists():
            continue
        log = json.loads(path.read_text())
        hist = log.get("history", [])
        evals = [h["eval_loss"] for h in hist if "eval_loss" in h] or [h["val_loss"] for h in hist if "val_loss" in h]
        where = log.get("gpu") or f"CPU, {log.get('cpu_threads', '?')} threads"
        rows.append(
            f"| {name} | {where} | {log.get('trainable_parameters', 0):,} ({log.get('trainable_pct')}%) | "
            f"{log.get('steps')} | {log.get('training_minutes')} min | "
            f"{evals[-1] if evals else '-'} |"
        )
    if len(rows) < 2:
        return ""
    head = ["| Run | Hardware | Trainable parameters | Steps | Training time | Final validation loss |",
            "|---|---|---|---|---|---|"]
    return "\n".join(["Training runs (from the logs saved next to each adapter):", ""] + head + rows)


def main() -> None:
    import markdown

    from evals import errors, final_table

    subprocess.run([sys.executable, "-m", "evals.final_table", "--write"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "evals.errors", "--write"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    del errors, final_table

    md_path = ROOT / "report" / "report.md"
    text = md_path.read_text()
    block = training_block()
    if block:
        text = re.sub(r"<!-- TRAINING:START -->.*?<!-- TRAINING:END -->",
                      lambda _: f"<!-- TRAINING:START -->\n{block}\n<!-- TRAINING:END -->", text, flags=re.DOTALL)
        md_path.write_text(text)
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)  # YAML front matter
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # block markers
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = ROOT / "report" / "report.html"
    html_path.write_text(html)

    chrome = next((c for c in CHROMES if c and Path(c).exists()), None)
    if not chrome:
        print("no Chrome/Chromium found: open report/report.html and print it to PDF")
        return
    pdf = ROOT / "report" / "report.pdf"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True)
    print("wrote", pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
