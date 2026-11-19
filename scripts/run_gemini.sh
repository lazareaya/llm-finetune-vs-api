#!/usr/bin/env bash
# Everything that needs the Gemini key, in one go:
#   1. Gemini Flash as the hosted comparison (200 test + 60 hand-written messages)
#   2. Gemini as the LLM judge (100 test messages, every system graded blind)
#   3. the final table, written into README.md and report/report.md
#
#   export GEMINI_API_KEY=<your Google AI Studio key>
#   bash scripts/run_gemini.sh
#
# Free-tier rate limits are handled by retries, and every answer is cached in evals/cache/,
# so if it stops halfway you can simply run it again: nothing is paid or asked twice.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export OPENAI_API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY:?set GEMINI_API_KEY first}}"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export OPENAI_MODEL="${OPENAI_MODEL:-gemini-3.8-flash}"
# Gemini 3 models always think a little before answering; "low" keeps that short, and the
# token cap must leave room for it (thinking tokens are billed as output and counted in the cost).
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-low}"
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-1024}"

python -m evals.run --systems hosted --set test --limit 200
python -m evals.run --systems hosted --set challenge
python -m evals.judge --set test --limit 100
python -m evals.final_table --write
