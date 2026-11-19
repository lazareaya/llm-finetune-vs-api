---
title: "Small fine-tuned LLM vs. large API model for structured extraction"
subtitle: "LoRA fine-tuning of Qwen3, LLM-as-a-judge, cost, latency and 4-bit quantisation on a customer-support triage task"
author: "lazareaya"
date: "November 2026"
---

# Small fine-tuned LLM vs. large API model for structured extraction

*LoRA fine-tuning of Qwen3, LLM-as-a-judge, cost, latency and 4-bit quantisation on a customer-support triage task.*
Code, data pipeline and raw outputs: <https://github.com/lazareaya/llm-finetune-vs-api>. Model: [LazareA/qwen3-0.6b-support-triage-lora](https://huggingface.co/LazareA/qwen3-0.6b-support-triage-lora).

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
Training runs (from the logs saved next to each adapter):

| Run | Hardware | Trainable parameters | Steps | Training time | Final validation loss |
|---|---|---|---|---|---|
| SmolLM2-135M (original, CPU) | CPU, 4 threads | 1,843,200 (1.35%) | 500 | 42.9 min | 0.0063 |
| qwen3-0.6b | Tesla T4 | 4,587,520 (0.76%) | 500 | 2.5 min | 0.0021 |
<!-- TRAINING:END -->

### 3.3 Hosted model

Gemini 3 Flash (`gemini-3-flash-preview`) is called through Google's OpenAI-compatible endpoint with the
same system prompt as the untrained models: the task, the full category → intents list and the order-number
rule. Temperature 0, JSON response mode. Gemini 3 models cannot turn thinking off, so
`reasoning_effort="low"` keeps it short. The thinking tokens are billed as output and are counted in the cost
(billed output = total tokens − prompt tokens).

*Free-tier constraint.* The free tier allows only 20 requests per day per Flash model, and the newest Flash
(`gemini-3.8-flash`) was saturated (HTTP 503) on the day of the run. So the accuracy runs send 20 to 25
numbered messages per request, with the same instructions plus "route each one independently, return one
record per number": 11 requests for 260 messages. Latency and cost per request are measured separately with
5 real single-message calls. This batching may change accuracy slightly compared with one message per
request. Re-running with `OPENAI_BATCH_SIZE=1` on a paid key (< $1 for the whole evaluation) removes it.

### 3.4 LLM-as-a-judge

Exact match is the right metric for routing, but it is all-or-nothing. An LLM judge can say *how wrong* an
answer is, and it is a common way to evaluate when there are no labels. To check whether it can be trusted
here, a different Gemini model (`gemini-3.5-flash`) grades every system on the first 100 test messages, from 1
(unusable) to 5 (same meaning as the reference). The judge gets a written rubric. To limit bias, all systems'
outputs for one message are graded together, under anonymous letters whose order is shuffled per message.
The judge never sees system names, and position bias averages out. Because of the daily quota, 10 messages
(70 outputs) are graded per request. Its grades are compared with exact match: agreement between
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
| SmolLM2-135M, instructions only | 135M | 0.0% (n=100) | 0.0% | 1.00 (n=100) | 480 ms (CPU, batched) | $0 API (+ ~$0.0067 compute) |
| SmolLM2-135M, instructions + 8 examples | 135M | 8.0% (n=100) | 100.0% | 2.99 (n=100) | 865 ms (CPU, batched) | $0 API (+ ~$0.0120 compute) |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 135M | 97.4% (n=1000) | 99.9% | 4.98 (n=100) | 766 ms (CPU, 1 request) | $0 API (+ ~$0.0106 compute) |
| Qwen3-0.6B, instructions only | 0.6B | 12.0% (n=100) | 100.0% | 3.03 (n=100) | 143 ms (T4 GPU, batched) | $0 API (+ ~$0.0140 compute) |
| **Qwen3-0.6B fine-tuned (LoRA, Unsloth)** | 0.6B | 99.7% (n=1000) | 100.0% | 5.00 (n=100) | 837 ms (T4 GPU, 16-bit, 1 request) | $0 API (+ ~$0.0814 compute) |
| **Qwen3-0.6B fine-tuned, 4-bit GGUF in Ollama (Mac)** | 0.6B (Q4_K_M, 396 MB) | 99.5% (n=200) | 100.0% | - | 203 ms (Mac 8 GB, Apple GPU (Metal), Ollama 4-bit, 1 request); 222 ms CPU only | $0 API (+ ~$0.0028 compute) |
| TF-IDF + logistic regression + regex | - | 95.5% (n=1000) | 100.0% | 4.95 (n=100) | 0.01 ms (CPU) | $0 API (+ ~$0.0000 compute) |
| Gemini 3 Flash (API), instructions only | large (undisclosed) | 94.0% (n=200) | 100.0% | 4.94 (n=100) | 2227 ms (API, free tier, 1 request) | $1.01 API (252 in + 294 out tokens/req) |

**Hand-written set** (60 messages in a different style, `evals/challenge.jsonl`)

| System | Whole record correct | Intent correct | Order number found | Valid JSON |
|---|---|---|---|---|
| SmolLM2-135M, instructions only | 0.0% | 0.0% | 0.0% | 0.0% |
| SmolLM2-135M, instructions + 8 examples | 1.7% | 16.7% | 6.2% | 98.3% |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 65.0% | 66.7% | 93.8% | 100.0% |
| Qwen3-0.6B, instructions only | 13.3% | 26.7% | 68.8% | 100.0% |
| **Qwen3-0.6B fine-tuned (LoRA, Unsloth)** | 81.7% | 81.7% | 100.0% | 100.0% |
| **Qwen3-0.6B fine-tuned, 4-bit GGUF in Ollama (Mac)** | 83.3% | 83.3% | 100.0% | 100.0% |
| TF-IDF + logistic regression + regex | 71.7% | 73.3% | 100.0% | 100.0% |
| Gemini 3 Flash (API), instructions only | 88.3% | 98.3% | 68.8% | 100.0% |

**Judge vs exact metrics** (gemini-3.5-flash-lite, 100 test messages, 700 graded outputs): grade 4-5 agrees with an exactly correct record 94.6% of the time (Cohen's kappa 0.886); mean grade 4.98 when exact, 2.21 when wrong; 2 exact records graded 1-3, 36 wrong records graded 4-5.

Cost: gemini-3-flash-preview: $0.5 input / $3.0 output per 1M tokens (https://aistudio.google.com/docs/pricing, checked 2026-11-19; thinking tokens are billed as output). Local compute is estimated at $0.05/h for a small CPU machine or laptop share and $0.35/h for a cloud T4 GPU, times the measured time per message (one machine, no parallelism).
<!-- RESULTS:END -->

<!-- FINDINGS:START -->
**What the numbers say**

1. **Fine-tuning turns a model that cannot do the task into one that beats the API on the task it was
   trained for.** Untouched, Qwen3-0.6B gets 12% of records right. After a 2.5-minute LoRA run on a free T4,
   it gets **99.7%** on 1,000 held-out messages. On the same 200 messages, it scores **99.5% against 94.0% for
   Gemini 3 Flash**, ahead of the 135M model (98.0%) and the classic classifier (96.5%).
2. **On unfamiliar wording, the large model still understands intent better.** On the 60 hand-written
   messages, Gemini gets 98.3% of intents right against 81.7% for the fine-tuned Qwen3. Its whole-record score
   (88.3%) is lower because it often drops the `#` from order numbers like `#48213`. With a lenient comparison
   that ignores the `#`, Gemini reaches 96.7%. The fine-tuned model copies the number exactly every time
   (100%), because it learned the exact output format.
3. **4-bit quantisation costs no accuracy here.** The q4_k_m GGUF (396 MB) scores 99.5% on the test sample and
   83.3% on the hand-written set (16-bit: 99.5% and 81.7%). Served by Ollama on an 8 GB MacBook, it answers
   in **203 ms** per request (222 ms on the CPU alone), against **2.2 s** for Gemini on the free tier.
4. **Cost.** Gemini 3 Flash costs about **$1.01 per 1,000 requests**: 252 input and 294 output tokens per
   request, most of the output being hidden thinking. The local model costs $0 in API fees and about $0.003
   of laptop time per 1,000 requests. At 1 million requests a month, that is about $1,000 against a few dollars
   of machine time, and the customer messages never leave the company.
5. **The LLM judge agrees with exact match (Cohen's kappa 0.89), but it is lenient on format.** It gives 4/5
   to records that use invented labels such as `"feedback"` or `"terminate_account"`. They mean the right
   thing, but routing code cannot use them. 34 of its 38 disagreements with exact match are records
   like these from the untrained models; 2 are Gemini records without the `#`; the fine-tuned models have none. A judge measures meaning; the guardrails and exact metrics measure whether the output can be
   used. Both are needed.
<!-- FINDINGS:END -->

## 5. Error analysis

<!-- ERRORS:START -->
Each wrong record counted once, in the first bucket that applies.

| System (test set) | n | Errors | no valid JSON | unknown label | wrong category | right category, wrong intent | order number missed | order number invented | order number wrong |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M, instructions only | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| SmolLM2-135M, instructions + 8 examples | 100 | 92 | 0 | 72 | 15 | 4 | 1 | 0 | 0 |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 1000 | 26 | 1 | 2 | 9 | 12 | 0 | 2 | 0 |
| Qwen3-0.6B, instructions only | 100 | 88 | 0 | 7 | 48 | 15 | 0 | 16 | 2 |
| **Qwen3-0.6B fine-tuned (LoRA, Unsloth)** | 1000 | 3 | 0 | 1 | 0 | 0 | 0 | 2 | 0 |
| **Qwen3-0.6B fine-tuned, 4-bit GGUF in Ollama (Mac)** | 200 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| TF-IDF + logistic regression + regex | 1000 | 45 | 0 | 0 | 4 | 5 | 0 | 36 | 0 |
| Gemini 3 Flash (API), instructions only | 200 | 12 | 0 | 0 | 3 | 3 | 0 | 4 | 2 |

| System (challenge set) | n | Errors | no valid JSON | unknown label | wrong category | right category, wrong intent | order number missed | order number invented | order number wrong |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M, instructions only | 60 | 60 | 60 | 0 | 0 | 0 | 0 | 0 | 0 |
| SmolLM2-135M, instructions + 8 examples | 60 | 59 | 1 | 49 | 9 | 0 | 0 | 0 | 0 |
| SmolLM2-135M fine-tuned (LoRA, original repo) | 60 | 21 | 0 | 1 | 13 | 6 | 1 | 0 | 0 |
| Qwen3-0.6B, instructions only | 60 | 52 | 0 | 4 | 32 | 9 | 0 | 5 | 2 |
| **Qwen3-0.6B fine-tuned (LoRA, Unsloth)** | 60 | 11 | 0 | 2 | 5 | 4 | 0 | 0 | 0 |
| **Qwen3-0.6B fine-tuned, 4-bit GGUF in Ollama (Mac)** | 60 | 10 | 0 | 3 | 4 | 3 | 0 | 0 | 0 |
| TF-IDF + logistic regression + regex | 60 | 17 | 0 | 0 | 12 | 4 | 0 | 1 | 0 |
| Gemini 3 Flash (API), instructions only | 60 | 7 | 0 | 0 | 1 | 0 | 0 | 1 | 5 |

Examples of errors, Qwen3-0.6B fine-tuned (LoRA, Unsloth), hand-written set:

- *Hi, first time here. How do I sign up so I can save my basket for later?* → expected `{"category": "ACCOUNT", "intent": "create_account", "order_number": null}`, got `{"category": "ORDER", "intent": "place_order", "order_number": null}` (wrong category)
- *The phone number on my account is the old one. Can I fix that myself?* → expected `{"category": "ACCOUNT", "intent": "edit_account", "order_number": null}`, got `{"category": "ACCOUNT", "intent": "switch_account", "order_number": null}` (right category, wrong intent)
- *What hours is your support team open, and what is the best way to reach them?* → expected `{"category": "CONTACT", "intent": "contact_customer_service", "order_number": null}`, got `{"category": "CONTACT", "intent": "contact_human_agent", "order_number": null}` (right category, wrong intent)

Examples of errors, Gemini 3 Flash (API), instructions only, hand-written set:

- *When should I expect order ORD-55120 to get here? I need it before the weekend.* → expected `{"category": "DELIVERY", "intent": "delivery_period", "order_number": "ORD-55120"}`, got `{"category": "ORDER", "intent": "track_order", "order_number": "ORD-55120"}` (wrong category)
- *Could you show me the invoice for order #48213? I need to check the VAT line.* → expected `{"category": "INVOICE", "intent": "check_invoice", "order_number": "#48213"}`, got `{"category": "INVOICE", "intent": "check_invoice", "order_number": "48213"}` (order number wrong)
- *Can I add one more item to order #903317, or swap the blue chair for the grey one?* → expected `{"category": "ORDER", "intent": "change_order", "order_number": "#903317"}`, got `{"category": "ORDER", "intent": "change_order", "order_number": "903317"}` (order number wrong)
<!-- ERRORS:END -->

<!-- ERROR-DISCUSSION:START -->
**Reading the error table.**

- *Fine-tuned Qwen3 on the test set*: 3 errors out of 1,000. Two are numbers glued to a word
  (`purchase00123842` → it extracted `order00123842`, which the dataset does not count as an order number).
  One is a misspelled label (`contact_customers_service`), which the guardrails reject and send to a person.
- *Fine-tuned Qwen3 on hand-written messages*: all 11 errors are about intent, none about the order number.
  The confusions are between neighbouring intents (`check_refund_policy` vs `get_refund`,
  `edit_account` vs `switch_account`, `track_order` vs `delivery_period`). Two outputs invent a label
  (`change_invoice`, `track_invoice`), and the guardrails catch them and send them to a person (3.3%; 5% for
  the 4-bit model, which also wrote `set_up_account`). The model learned
  the dataset's phrasing more than the meaning of the labels. More varied training messages would help more
  than a bigger rank or more epochs.
- *Gemini on hand-written messages*: one real intent confusion (`track_order` for "when should I expect
  order X", which is arguably correct). One invoice number taken for an order number. Five order numbers
  without their `#`. These are format errors, not understanding errors. Adding examples to the prompt, or
  normalising order numbers in code, would fix most of them, at the cost of a longer and more expensive
  prompt.
- *Gemini on the test set*: most errors follow the dataset's own conventions, not a misunderstanding. The
  dataset says `bill #12588` is not an order number and that "when will my item arrive" is
  `delivery_period`, not `track_order`. A fine-tuned model learns these conventions from 8,000 examples; a
  prompted model has to guess them.
- *Classic classifier*: very strong on intent, but the regular expression takes any number. 36 of its
  45 test errors are invented order numbers (phone numbers, invoice numbers).
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
- **Judge bias and strength.** The judge (`gemini-3.5-flash-lite`) is from the same family as the compared
  hosted model, so self-preference is possible. Blinding and shuffling reduce position and name bias, not style
  bias. It is a "lite" model, chosen because the stronger Flash models were saturated or out of free quota. A
  stronger judge might be stricter on invented labels. The judge was checked against exact labels, which is
  only possible because labels exist.
- **Hosted model run in batches, on a smaller sample.** Because of the free-tier quota, Gemini answered 20 to
  25 messages per request, and only 200 of the 1,000 test messages were used. The fine-tuned model's
  advantage is compared on the same 200 messages (99.5% vs 94.0%), but 200 messages give about ±3 points of
  uncertainty on Gemini's score.
- **Different hardware for latency.** Batched throughput was measured on a CPU, an Apple GPU or a T4,
  depending on the model. The table states the hardware for every number. Hosted latency includes the
  network and Google's queueing on the free tier.
- **Prices change.** Gemini list prices were checked on 19 November 2026 (`gemini-3-flash-preview`: $0.50 /
  $3.00 per million input/output tokens). The newest Flash models cost $0.75 / $3.75 and double on 1 January
  2027, so the hosted cost here is a lower bound. Free-tier latency includes Google's queueing on a busy day.
- **English only**, one order number per message, and only the order number is extracted.

## 7. Conclusion

<!-- CONCLUSION:START -->
For a narrow extraction task with a fixed schema and enough labelled examples, **a 0.6B model
fine-tuned with LoRA in a few minutes on a free GPU is more accurate than a large hosted model on in-domain
traffic** (99.5% vs 94.0% on the same 200 messages). Quantised to 4 bits, it is about **10 times faster**
(0.2 s vs 2.2 s), **free per request** (vs about $1 per 1,000) and **private**. The large model's advantage
is robustness: on messages written differently from the training data, it understands intent better (98.3%
vs 81.7%). It needs no training data at all, but it does not follow output conventions it was not shown.

Practical recommendation: use the fine-tuned small model behind the guardrails for the bulk of the traffic.
Send what the guardrails reject (3-5% on unfamiliar wording) to the large model or to a person, and use
the large model's answers on hard cases to grow the training set. The LLM judge is useful for monitoring
quality without labels, but it should not replace exact checks on structured output: it rewards answers
that are right in meaning but unusable.
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
