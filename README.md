# Small fine-tuned LLM vs. large API model — support-ticket triage

**Question:** for a narrow, structured task, can a small open LLM fine-tuned with LoRA match a large
hosted model (Gemini Flash), and at what cost, latency and reliability?

**Task:** read a customer-support message and write a JSON routing record:

```json
{"category": "REFUND", "intent": "get_refund", "order_number": "PX300127"}
```

11 categories, 27 intents, and an order number that must be copied exactly (or `null`).
Data: the public [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(26,872 messages), plus 60 hand-written messages in a different style.

| | |
|---|---|
| Fine-tuned model (LoRA adapter) | HF_MODEL_LINK |
| 4-bit GGUF for Ollama / llama.cpp | HF_GGUF_LINK |
| Technical report (EN) | [report/report.pdf](report/report.pdf) · [Markdown](report/report.md) |
| Colab training notebook (Unsloth) | [notebooks/finetune_qwen3_unsloth.ipynb](notebooks/finetune_qwen3_unsloth.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lazareaya/llm-finetune-vs-api/blob/main/notebooks/finetune_qwen3_unsloth.ipynb) |

## Results

Every system is scored by the same code on the same messages (`evals/run.py`), graded blind by an
LLM judge (`evals/judge.py`), and priced per 1,000 requests (`evals/final_table.py`).
All raw outputs are in [`evals/outputs/`](evals/outputs) and all numbers in [`evals/report.json`](evals/report.json).

<!-- RESULTS:START -->
**Held-out test set** (public dataset messages never used for training)

| System | Size | Whole record correct | Valid JSON | Judge grade (1-5) | Latency per message | Cost per 1,000 requests |
|---|---|---|---|---|---|---|
| SmolLM2-135M, instructions only | 135M | 0.0% (n=100) | 0.0% | - | 480 ms (CPU, batched) | $0 API (+ ~$0.0067 compute) |
| SmolLM2-135M, instructions + 8 examples | 135M | 8.0% (n=100) | 100.0% | - | 865 ms (CPU, batched) | $0 API (+ ~$0.0120 compute) |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 135M | 97.4% (n=1000) | 99.9% | - | 766 ms (CPU, 1 request) | $0 API (+ ~$0.0106 compute) |
| Qwen3-0.6B, instructions only | 0.6B | 14.0% (n=100) | 100.0% | - | 936 ms (Apple GPU, batched) | $0 API (+ ~$0.0130 compute) |
| TF-IDF + logistic regression + regex | - | 95.5% (n=1000) | 100.0% | - | 0.01 ms (CPU) | $0 API (+ ~$0.0000 compute) |

**Hand-written set** (60 messages in a different style, `evals/challenge.jsonl`)

| System | Whole record correct | Intent correct | Order number found | Valid JSON |
|---|---|---|---|---|
| SmolLM2-135M, instructions only | 0.0% | 0.0% | 0.0% | 0.0% |
| SmolLM2-135M, instructions + 8 examples | 1.7% | 16.7% | 6.2% | 98.3% |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 65.0% | 66.7% | 93.8% | 100.0% |
| Qwen3-0.6B, instructions only | 15.0% | 28.3% | 68.8% | 100.0% |
| TF-IDF + logistic regression + regex | 71.7% | 73.3% | 100.0% | 100.0% |

Cost: Gemini Flash list price (https://aistudio.google.com/docs/pricing, checked 2026-11-19; thinking tokens are billed as output). Local compute is estimated at $0.05/h for a small CPU machine or laptop share and $0.35/h for a cloud T4 GPU, times the measured time per message (one machine, no parallelism).
<!-- RESULTS:END -->

<!-- FINDINGS:START -->
<!-- FINDINGS:END -->

## How it works

```mermaid
flowchart LR
    A[Bitext dataset<br/>26,872 messages] --> B[Data script<br/>placeholders, dedupe,<br/>split, augmentation]
    B --> C[8,000 training pairs<br/>message → JSON]
    C --> D[LoRA fine-tuning<br/>Unsloth + TRL on a Colab T4<br/>rank 16, 1 epoch]
    E[Qwen3-0.6B / 1.7B] --> D
    D --> F[LoRA adapter<br/>→ Hugging Face Hub]
    D --> Q[Merged + 4-bit GGUF q4_k_m<br/>→ Ollama on a Mac]
    M[Customer message] --> G[Model]
    G --> H[Generated JSON]
    H --> I{Guardrails in code<br/>valid JSON, known labels,<br/>intent ∈ category,<br/>order number in message}
    I -- pass --> J[Route automatically]
    I -- fail --> K[Send to a person]
    H --> L[Evaluation: exact metrics<br/>+ Gemini as blind judge<br/>+ latency + cost]
```

- **LoRA**: the base model is frozen; rank-16 adapters on the attention projections are trained
  (0.76% of Qwen3-0.6B's weights). Only the JSON answer carries a loss, never the customer message.
- **Guardrails**: output is parsed and validated (Pydantic), labels must be on the approved list,
  the intent must belong to the category, and the order number must literally appear in the message.
  Anything that fails goes to a human.
- **Hosted baseline**: Gemini Flash through its OpenAI-compatible endpoint, with the full label list
  in the prompt and JSON mode; token usage (including thinking tokens) is recorded for the cost column.
- **LLM-as-a-judge**: for each message, Gemini sees the reference record and all systems' outputs under
  shuffled anonymous letters and grades each 1-5; the grades are then compared with the exact metrics
  (agreement, Cohen's kappa, disagreement cases).
- **Quantisation & serving**: the merged model is exported to GGUF q4_k_m and served by Ollama on a
  laptop; latency is measured one request at a time.

## What I added (compared with the original project)

The starting point ([BelAnas/small-llm-lora-finetune](https://github.com/BelAnas/small-llm-lora-finetune),
original README kept in [docs/original_README.md](docs/original_README.md)) fine-tuned a 135M model on a CPU
and compared it with prompting and a classic classifier. In this fork I:

1. **Fine-tuned a real LLM on GPU**: Qwen3-0.6B (and 1.7B) with **Unsloth + TRL + PEFT** on a Colab T4
   ([notebook](notebooks/finetune_qwen3_unsloth.ipynb)), adapter published on the Hugging Face Hub.
   The repo's own training code also runs on CUDA / Apple GPUs now (device selection, mixed precision,
   gradient accumulation).
2. **Measured a large hosted model**: Gemini Flash via the OpenAI-compatible API (retries for free-tier rate
   limits, response cache, thinking-token accounting).
3. **Added an LLM judge** (`evals/judge.py`): blind, order-shuffled, one call per message for all
   systems, and an agreement analysis against the exact metrics.
4. **Added cost and latency analysis** (`evals/final_table.py`): cost per 1,000 requests from measured
   tokens and list prices, compute estimate for local models, per-request latency.
5. **Quantised the model to 4 bits** (GGUF q4_k_m) and **served it locally with Ollama**
   (`OllamaProvider`, `scripts/ollama_create.sh`), with latency measured on a Mac.
6. **Error analysis** by error type (`evals/errors.py`) and a **technical report** ([report/](report/)).
7. Tests for all of the above (`tests/test_additions.py`, 60 offline tests in total).

## Reproduce

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
pytest -q                                                    # 60 offline tests
python -m triage.cli data                                    # dataset → data/{train,val,test}.jsonl

# Local baselines (CPU)
python -m evals.run --systems classic lora mock --set test
python -m evals.run --systems base-prompt base-examples --set test --limit 100

# Qwen3 fine-tuning + evaluation + GGUF: open the notebook in Colab (T4), Run all, then:
python scripts/fetch_colab_results.py <hf-user>/qwen3-0.6b-support-triage-lora
bash scripts/ollama_create.sh qwen3-0.6b                     # registers the 4-bit model in Ollama
OLLAMA_MODEL=triage-qwen3-0.6b python -m evals.latency --system ollama
OLLAMA_MODEL=triage-qwen3-0.6b python -m evals.run --systems ollama --set test --limit 200

# Gemini comparison + LLM judge + final tables (free Google AI Studio key)
export GEMINI_API_KEY=...
bash scripts/run_gemini.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `TRIAGE_PROVIDER` | `local` | API/CLI backend: `local`, `mock`, `openai` (any OpenAI-compatible API, e.g. Gemini), `ollama` |
| `TRIAGE_DEVICE` | auto | `cpu`, `cuda` or `mps` |
| `TRIAGE_ADAPTER_QWEN3_0_6B` | `adapters/qwen3-0.6b` | local folder or Hub repo id of the Qwen3 adapter |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | – | hosted model; for Gemini use `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `OPENAI_REASONING_EFFORT`, `OPENAI_MAX_TOKENS` | –, 60 | thinking level and token cap for thinking models |
| `OLLAMA_MODEL`, `OLLAMA_URL` | `triage-qwen3-0.6b`, `http://127.0.0.1:11434` | local 4-bit model |

The API: `python -m triage.cli serve`, then `POST /route {"message": "..."}` (docs at `/docs`).

## Limits

See the [report](report/report.md) (section 6) for the full list. In short: the public dataset is
template-generated, so held-out scores flatter every trained system; the hand-written set is small (60)
and was written by the original author; there is no "none of these" label; one seed per configuration;
the judge is from the same family as one of the compared systems (Gemini), a known self-preference
risk; prices are list prices on the date checked.

## Credits

Based on [BelAnas/small-llm-lora-finetune](https://github.com/BelAnas/small-llm-lora-finetune) (MIT):
the task, data pipeline, guardrails, scorer, classic baseline and the 135M CPU run are theirs.
Training notebook adapted from [Unsloth's notebooks](https://github.com/unslothai/notebooks).
Base models: [Qwen3](https://huggingface.co/Qwen) (Apache 2.0), [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Apache 2.0).
Data: Bitext customer-support dataset (CDLA-Sharing-1.0), downloaded by the data script and not stored here.
Licence of this repository: MIT, see [LICENSE](LICENSE).
