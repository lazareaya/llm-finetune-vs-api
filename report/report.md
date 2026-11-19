---
title: "Small fine-tuned LLM vs. large API model for structured extraction"
subtitle: "LoRA fine-tuning of Qwen3, LLM-as-a-judge, cost, latency and 4-bit quantisation on a customer-support triage task"
author: "lazareaya"
date: "November 2026"
---

# Small fine-tuned LLM vs. large API model for structured extraction

*LoRA fine-tuning of Qwen3, LLM-as-a-judge, cost, latency and 4-bit quantisation on a customer-support triage task.*
Code, data pipeline and raw outputs: <https://github.com/lazareaya/llm-finetune-vs-api>. Model: HF_MODEL_LINK.

## 1. Question

Teams that route customer messages automatically have two cheap options today: call a large hosted
model with a good prompt, or fine-tune a small open model and run it themselves. The hosted model needs
no training but is paid per request, depends on a third party and sends customer text outside the company.
The small model is free per request and private, but it has to be trained and served.

This project measures the trade-off on one narrow, structured task, with the same messages, the same
checks and the same scorer for every system. The questions are:

1. How close does a fine-tuned 0.6B-1.7B model get to a large hosted model (Gemini Flash) on accuracy?
2. Does an LLM judge agree with exact metrics on a task like this, and what does it add?
3. What does each option cost per 1,000 requests, and how fast is a 4-bit model on a laptop?

## 2. Task and data

A system reads one customer message and writes one JSON record:
`{"category": ..., "intent": ..., "order_number": ...}` with 11 categories, 27 intents (each intent belongs to
exactly one category) and an order number that must be copied exactly from the message, or `null`.

**Data.** The public Bitext customer-support dataset (26,872 messages, 27 intents, CDLA-Sharing-1.0).
The data script (inherited from the original project) fills the dataset's placeholders (`{{Order Number}}`…)
with invented values in six different formats, removes duplicate messages (24,274 unique), and makes a
seeded split: 8,000 training, 300 validation and 1,000 test messages. In the source data an order number
only appears with a few intents, so 15% of training messages get an order-number mention added and 6% an
unrelated number (phone, postcode, customer ID) as a distractor; validation and test are left untouched.

**Two evaluation sets.** (1) The 1,000 held-out test messages. Because Bitext was generated from templates,
these resemble the training messages and flatter every trained system. (2) 60 hand-written messages in a
different style (`evals/challenge.jsonl`): typos, indirect wording, extra numbers. This is the harder,
more realistic set, but small: one message is 1.7 points.

**Metrics.** A record is *valid* if it parses and matches the schema; *whole record correct* (the main
metric) if category, intent and order number are all right. The guardrails used in production also run in
evaluation: unknown labels, an intent that does not belong to its category, or an order number that does
not appear in the message send the record to a human.

## 3. Method

### 3.1 Systems compared

| System | What it is |
|---|---|
| SmolLM2-135M, prompt / + 8 examples | Original project's base model with the full label list in the prompt (zero-shot / few-shot) |
| SmolLM2-135M + LoRA | Original project's adapter, trained on a CPU (43 min) |
| Qwen3-0.6B, prompt | Untouched Qwen3 with the same instructions (thinking disabled) |
| **Qwen3-0.6B + LoRA** (and 1.7B) | Fine-tuned in this work with Unsloth on a Colab T4 |
| Qwen3-0.6B + LoRA, 4-bit GGUF | Same model, merged and quantised to q4_k_m, served by Ollama on a Mac |
| TF-IDF + logistic regression + regex | Classic baseline; a regex picks any 4+ digit token as the order number |
| Gemini Flash (API) | Large hosted model, same instructions, JSON mode, via the OpenAI-compatible endpoint |

### 3.2 Fine-tuning

Qwen3 is loaded in 16-bit with Unsloth and LoRA adapters (rank 16, alpha 32) are added to the query, key,
value and output projections of every attention layer, the same layers as the original 135M run. That is
4.6M trainable parameters for Qwen3-0.6B, or 0.76% of the model. Each training example is
`### Message\n<message>\n### Record\n<json><|im_end|>`, and the prompt tokens are masked out of the loss,
so only the record and the stop token are learned. The repository's own `encode_example` is reused in the
notebook, so the masking is identical to the CPU run. Training uses TRL's `SFTTrainer`: 1 epoch over 8,000
examples (500 steps of 16), learning rate 2e-4, 30 warm-up steps then cosine decay, 8-bit AdamW, seed 7.
The short prompt is deliberate: after fine-tuning the model no longer needs the label list, so a request
is the message plus about 8 tokens of template, against about 200 tokens of instructions for the hosted model.

<!-- TRAINING:START -->
<!-- TRAINING:END -->

### 3.3 Hosted model

Gemini Flash is called through Google's OpenAI-compatible endpoint with the same system prompt as the
untrained models: the task, the full category → intents list and the order-number rule. Temperature 0,
JSON response mode. Gemini 3 models cannot turn thinking off, so `reasoning_effort="low"` keeps it short.
The thinking tokens are billed as output, and they are counted in the cost (billed output = total tokens −
prompt tokens). Calls are retried on rate limits and cached, so the free tier is enough to run the evaluation.

### 3.4 LLM-as-a-judge

Exact match is the right metric for routing, but it is all-or-nothing. An LLM judge can say *how wrong* an
answer is, and it is a common way to evaluate when there are no labels. To check whether it can be trusted
here, Gemini grades every system on the first 100 test messages, from 1 (unusable) to 5 (same meaning as the
reference). The judge gets a written rubric. To limit bias, all systems' outputs for one message go in a
single request, under anonymous letters whose order is shuffled per message. The judge never sees system
names, and position bias averages out. Its grades are compared with exact match: agreement between
"grade ≥ 4" and "exactly correct", Cohen's kappa, and the cases where they disagree.

### 3.5 Quantisation, serving and cost

The fine-tuned Qwen3 is merged with its adapter and exported by Unsloth to GGUF with q4_k_m quantisation,
which takes the 0.6B model from about 1.2 GB in 16-bit to about 0.4 GB. It is served by Ollama on a MacBook
(Apple M-series, 8 GB) in raw mode, so it gets exactly the training prompt. Latency is measured one request
at a time after a warm-up, as an API would call it.

The cost per 1,000 requests is computed from measured tokens and list prices for the hosted model. For local
models it is $0 in API fees plus an estimate of machine time: measured seconds per request times an hourly
price ($0.05/h for a CPU share, $0.35/h for a cloud T4).

## 4. Results

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

## 5. Error analysis

<!-- ERRORS:START -->
Each wrong record counted once, in the first bucket that applies.

| System (test set) | n | Errors | no valid JSON | unknown label | wrong category | right category, wrong intent | order number missed | order number invented | order number wrong |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M, instructions only | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| SmolLM2-135M, instructions + 8 examples | 100 | 92 | 0 | 72 | 15 | 4 | 1 | 0 | 0 |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 1000 | 26 | 1 | 2 | 9 | 12 | 0 | 2 | 0 |
| Qwen3-0.6B, instructions only | 100 | 86 | 0 | 7 | 47 | 14 | 0 | 16 | 2 |
| TF-IDF + logistic regression + regex | 1000 | 45 | 0 | 0 | 4 | 5 | 0 | 36 | 0 |

| System (challenge set) | n | Errors | no valid JSON | unknown label | wrong category | right category, wrong intent | order number missed | order number invented | order number wrong |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M, instructions only | 60 | 60 | 60 | 0 | 0 | 0 | 0 | 0 | 0 |
| SmolLM2-135M, instructions + 8 examples | 60 | 59 | 1 | 49 | 9 | 0 | 0 | 0 | 0 |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 60 | 21 | 0 | 1 | 13 | 6 | 1 | 0 | 0 |
| Qwen3-0.6B, instructions only | 60 | 51 | 0 | 4 | 31 | 9 | 0 | 5 | 2 |
| TF-IDF + logistic regression + regex | 60 | 17 | 0 | 0 | 12 | 4 | 0 | 1 | 0 |

Examples of errors, SmolLM2-135M fine-tuned (LoRA, original repo), hand-written set:

- *Hi, first time here. How do I sign up so I can save my basket for later?* → expected `{"category": "ACCOUNT", "intent": "create_account", "order_number": null}`, got `{"category": "ORDER", "intent": "place_order", "order_number": null}` (wrong category)
- *Please erase my profile and everything attached to it, I am not going to shop here again.* → expected `{"category": "ACCOUNT", "intent": "delete_account", "order_number": null}`, got `{"category": "ACCOUNT", "intent": "recover_password", "order_number": null}` (right category, wrong intent)
- *I moved to a new job and my email changed, where do I update the details on my profile?* → expected `{"category": "ACCOUNT", "intent": "edit_account", "order_number": null}`, got `{"category": "FEEDBACK", "intent": "review", "order_number": null}` (wrong category)
- *The phone number on my account is the old one. Can I fix that myself?* → expected `{"category": "ACCOUNT", "intent": "edit_account", "order_number": null}`, got `{"category": "ACCOUNT", "intent": "recover_password", "order_number": null}` (right category, wrong intent)
<!-- ERRORS:END -->

<!-- ERROR-DISCUSSION:START -->
<!-- ERROR-DISCUSSION:END -->

## 6. Limitations

- **Template data.** Bitext was generated from templates. Test messages look like training messages, so
  test-set accuracy overstates real-world accuracy for every trained system. The hand-written set is the
  more honest number, but it has only 60 messages: a 5-point gap is 3 messages and not significant.
- **Hand-written set not fully blind.** The original author wrote it and tuned the data augmentation after
  seeing results on it (documented in `docs/original_README.md`). This work did not tune anything on it.
- **No "none of these" label.** An off-topic message is forced into the nearest intent, and the guardrails
  cannot catch it because the record is well formed. A deployment needs an `other` class.
- **One run per configuration**, one seed, no hyper-parameter search, no confidence intervals. Differences of
  about 1 point on the test set should not be over-read.
- **Judge bias.** The judge is a Gemini model and one of the compared systems is Gemini, so self-preference
  is possible. Blinding and shuffling reduce position and name bias, not style bias. The judge is checked
  against exact labels here, which is only possible because labels exist.
- **Different hardware for latency.** Batched throughput was measured on a CPU, an Apple GPU or a T4,
  depending on the model. The table states the hardware for every number. Hosted latency includes the
  network and Google's queueing on the free tier.
- **Prices change.** Gemini list prices were checked on 19 November 2026. Google has announced that
  Flash prices double on 1 January 2027, which would double the hosted cost column.
- **English only**, one order number per message, and only the order number is extracted.

## 7. Conclusion

<!-- CONCLUSION:START -->
<!-- CONCLUSION:END -->

## Appendix: reproduction

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
python -m triage.cli data && pytest -q
python -m evals.run --systems classic lora --set test           # and --set challenge
# Colab: notebooks/finetune_qwen3_unsloth.ipynb (T4, Run all), then
python scripts/fetch_colab_results.py <hf-user>/qwen3-0.6b-support-triage-lora
bash scripts/ollama_create.sh qwen3-0.6b && OLLAMA_MODEL=triage-qwen3-0.6b python -m evals.latency --system ollama
GEMINI_API_KEY=... bash scripts/run_gemini.sh                   # hosted model, judge, final tables
python -m evals.errors --write && python scripts/build_report.py
```

Based on [BelAnas/small-llm-lora-finetune](https://github.com/BelAnas/small-llm-lora-finetune) (MIT), which
provides the task, the data pipeline, the guardrails, the scorer, the classic baseline and the 135M CPU run.
