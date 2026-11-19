"""Settings read from the environment, so nothing is hard-coded in the modules that use them."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# The larger open models fine-tuned in this fork: system name -> (base model, default adapter).
# The adapter is a local folder or a Hugging Face Hub repo id; override it with
# TRIAGE_ADAPTER_QWEN3_0_6B / TRIAGE_ADAPTER_QWEN3_1_7B.
QWEN_MODELS: dict[str, tuple[str, str]] = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "adapters/qwen3-0.6b"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "adapters/qwen3-1.7b"),
}


def resolve_adapter(value: str | Path) -> Path | str:
    """A local folder (relative to the repo) or, when no such folder exists, a Hub repo id."""
    path = Path(value)
    if path.is_absolute():
        return path
    local = ROOT / path
    if local.exists() or str(value).count("/") != 1:
        return local
    return str(value)


def qwen_model(name: str) -> tuple[str, Path | str]:
    base, adapter = QWEN_MODELS[name]
    env = "TRIAGE_ADAPTER_" + name.upper().replace("-", "_").replace(".", "_")
    return base, resolve_adapter(os.getenv(env, adapter))


@dataclass(frozen=True)
class Settings:
    provider: str
    base_model: str
    adapter_dir: Path | str
    threads: int
    openai_model: str
    ollama_model: str
    ollama_url: str
    max_new_tokens: int = 48  # the longest valid record is about 35 tokens


def get_settings() -> Settings:
    return Settings(
        provider=os.getenv("TRIAGE_PROVIDER", "local").lower(),
        base_model=os.getenv("TRIAGE_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        adapter_dir=resolve_adapter(os.getenv("TRIAGE_ADAPTER_DIR", "adapter")),
        threads=int(os.getenv("TRIAGE_THREADS", "0")),
        openai_model=os.getenv("OPENAI_MODEL", "") or "gpt-4o-mini",
        ollama_model=os.getenv("OLLAMA_MODEL", "triage-qwen3-0.6b"),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
    )
