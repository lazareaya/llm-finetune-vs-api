"""The systems that can route a message. Each takes messages and returns raw text per message.

They all return text in the same format, so the checks in schema.py and the scorer in
metrics.py treat a rule, a small local model and a hosted model in exactly the same way.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Protocol

from .baseline import regex_order_number
from .config import Settings
from .labels import INTENT_TO_CATEGORY
from .prompts import chat_messages, ft_prompt
from .schema import target_json


class Provider(Protocol):
    name: str

    def complete(self, messages: list[str]) -> list[str]: ...


# Ordered keyword rules: the first one that matches wins, so specific rules come first.
_RULES: list[tuple[str, str]] = [
    (r"refund.*(status|where|track|still waiting)|track.*refund", "track_refund"),
    (r"refund polic|money back guarantee|conditions.*refund", "check_refund_policy"),
    (r"refund|reimburs|money back", "get_refund"),
    (r"password|pass ?word|locked out", "recover_password"),
    (r"human|real person|live agent|speak (to|with) (a|an|some)", "contact_human_agent"),
    (r"customer (service|support|care)|support team", "contact_customer_service"),
    (r"newsletter|mailing list", "newsletter_subscription"),
    (r"cancel\w* (fee|charge|penalt)|penalt", "check_cancellation_fee"),
    (r"invoice|bill\b", "get_invoice"),
    (r"(change|update|correct|wrong).*address", "change_shipping_address"),
    (r"address", "set_up_shipping_address"),
    (r"cancel", "cancel_order"),
    (r"where is|track|status of", "track_order"),
    (r"(add|remove|swap|change).*(order|item)", "change_order"),
    (r"payment (method|option)|ways of paying|accept", "check_payment_methods"),
    (r"payment|charged|card", "payment_issue"),
    (r"complain|unhappy|worst", "complaint"),
    (r"feedback|review", "review"),
    (r"(how long|when).*(deliver|arriv|ship)", "delivery_period"),
    (r"deliver|shipping", "delivery_options"),
    (r"(delete|close|erase).*(account|profile)", "delete_account"),
    (r"(sign.?up|regist).*(error|problem|fail)", "registration_problems"),
    (r"sign.?up|register|create.*(account|profile)", "create_account"),
    (r"(switch|upgrade|move).*(account|plan)", "switch_account"),
    (r"account|profile", "edit_account"),
    (r"buy|order|purchase", "place_order"),
]


class MockProvider:
    """Keyword rules, no model. It exists so the tests and the API run with nothing downloaded.

    It is not a baseline anyone should ship: its accuracy is reported only to make clear
    that it is a stand-in.
    """

    name = "mock"

    def complete(self, messages: list[str]) -> list[str]:
        out = []
        for message in messages:
            text = message.lower()
            intent = next((i for pattern, i in _RULES if re.search(pattern, text)), "contact_customer_service")
            out.append(target_json(INTENT_TO_CATEGORY[intent], intent, regex_order_number(message)))
        return out


class LocalProvider:
    """A model on this machine: the fine-tuned one, or the base model driven by a prompt."""

    def __init__(
        self,
        model,
        tokenizer,
        mode: str = "lora",
        examples: list[dict] | None = None,
        max_new_tokens: int = 48,
        batch_size: int = 16,
        name: str | None = None,
    ):
        self.model, self.tokenizer, self.mode = model, tokenizer, mode
        self.examples = examples or []
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.name = name or {"lora": "lora", "zero": "base-prompt", "few": "base-examples"}[mode]

    def prompt(self, message: str) -> str:
        from .model import render_chat

        if self.mode == "lora":
            return ft_prompt(message)
        return render_chat(self.tokenizer, chat_messages(message, self.examples if self.mode == "few" else None))

    def complete(self, messages: list[str]) -> list[str]:
        from .model import generate

        prompts = [self.prompt(m) for m in messages]
        return generate(self.model, self.tokenizer, prompts, self.max_new_tokens, self.batch_size)


class OpenAIProvider:
    """A hosted model with the same instructions, for a cost and accuracy comparison.

    Any OpenAI-compatible endpoint works: OpenAI itself, or Google Gemini with
    OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ (read by the client).
    """

    name = "hosted"

    def __init__(
        self,
        model: str,
        client=None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        cache_path: Path | None = None,
        retries: int = 10,
    ):
        if client is None:
            from openai import OpenAI

            # Reads OPENAI_API_KEY and OPENAI_BASE_URL; retries are handled below, with a cap per call.
            client = OpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT", "90")), max_retries=0)
        self.client, self.model = client, model
        # Models that think before answering spend output tokens on it, so they need a larger cap.
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "60"))
        self.reasoning_effort = reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT", "")
        self.cache_path, self.retries = cache_path, retries
        self.input_tokens = self.output_tokens = 0
        self.latencies_ms: list[float] = []
        self.rate_limited = 0
        self._last_ms = 0.0
        self._cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            for line in Path(cache_path).read_text().splitlines():
                row = json.loads(line)
                self._cache[row["key"]] = row

    def _call(self, message: str):
        kwargs = dict(
            model=self.model,
            messages=chat_messages(message),
            max_tokens=self.max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        for attempt in range(self.retries + 1):
            try:
                started = time.perf_counter()
                response = self.client.chat.completions.create(**kwargs)
                # Latency of the successful call only: waiting for a free-tier rate limit is not latency.
                self._last_ms = 1000 * (time.perf_counter() - started)
                return response
            except Exception as exc:  # rate limits on free tiers: wait and try again
                status = getattr(exc, "status_code", None)
                if status == 429:
                    self.rate_limited += 1
                # status None = timeout or connection error: also worth another try.
                if attempt == self.retries or status not in (None, 429, 500, 502, 503, 504):
                    raise
                time.sleep(min(60, 5 * 2**attempt))

    def complete(self, messages: list[str]) -> list[str]:
        out = []
        for message in messages:
            key = f"{self.model}|{self.reasoning_effort}|{message}"
            row = self._cache.get(key)
            if row is None:
                response = self._call(message)
                ms = self._last_ms
                usage = getattr(response, "usage", None)
                prompt_t = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
                # Billed output includes any hidden reasoning tokens: total minus prompt.
                total_t = getattr(usage, "total_tokens", None) if usage else None
                completion_t = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
                output_t = max(completion_t, (total_t or 0) - prompt_t) if isinstance(total_t, int) else completion_t
                row = {
                    "key": key,
                    "text": response.choices[0].message.content or "",
                    "input_tokens": prompt_t,
                    "output_tokens": output_t,
                    "ms": round(ms, 1),
                }
                self._cache[key] = row
                if self.cache_path:
                    Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with Path(self.cache_path).open("a") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.input_tokens += row["input_tokens"]
            self.output_tokens += row["output_tokens"]
            self.latencies_ms.append(row["ms"])
            out.append(row["text"])
        return out


class OllamaProvider:
    """The fine-tuned model, quantised to 4 bits (GGUF) and served by Ollama on this machine.

    The fine-tuned model was trained on a plain prompt, not a chat template, so the request
    uses raw mode: Ollama sends exactly ft_prompt(message) to the model.
    """

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", max_new_tokens: int = 48):
        import httpx

        self.name = f"ollama-{model}"
        self.model, self.max_new_tokens = model, max_new_tokens
        self.http = httpx.Client(base_url=url, timeout=120)
        self.latencies_ms: list[float] = []

    def complete(self, messages: list[str]) -> list[str]:
        out = []
        for message in messages:
            started = time.perf_counter()
            response = self.http.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": ft_prompt(message),
                    "raw": True,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0, "num_predict": self.max_new_tokens},
                },
            )
            response.raise_for_status()
            self.latencies_ms.append(1000 * (time.perf_counter() - started))
            out.append(response.json().get("response", "").strip())
        return out


def pick_examples(train: list[dict], k: int = 8, seed: int = 7) -> list[dict]:
    """Worked examples for the prompt: different intents, a quarter of them with an order number."""
    rng = random.Random(seed)
    pool = train[:]
    rng.shuffle(pool)
    want_order = max(1, k // 4)
    picked: list[dict] = []
    seen: set[str] = set()
    for ex in pool:
        need_order = sum(1 for p in picked if p["order_number"]) < want_order
        if ex["intent"] in seen or (need_order and not ex["order_number"]):
            continue
        if not need_order and ex["order_number"]:
            continue
        picked.append(ex)
        seen.add(ex["intent"])
        if len(picked) == k:
            break
    return picked


def build_provider(settings: Settings) -> Provider:
    """The provider named in the settings. Used by the API and the command line."""
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "openai":
        return OpenAIProvider(settings.openai_model)
    if settings.provider == "ollama":
        return OllamaProvider(settings.ollama_model, settings.ollama_url, settings.max_new_tokens)
    if settings.provider == "local":
        from .model import has_adapter, load_model, load_tokenizer, pick_device, set_threads

        if not has_adapter(settings.adapter_dir):
            raise RuntimeError(
                f"No LoRA add-on in {settings.adapter_dir}. Run `python -m triage.cli train` first, "
                "or set TRIAGE_PROVIDER=mock."
            )
        set_threads(settings.threads)
        tokenizer = load_tokenizer(settings.base_model)
        model = load_model(settings.base_model, settings.adapter_dir, merge=True, device=pick_device())
        return LocalProvider(model, tokenizer, "lora", max_new_tokens=settings.max_new_tokens)
    raise ValueError(f"Unknown TRIAGE_PROVIDER '{settings.provider}' (use local, mock, openai or ollama)")
