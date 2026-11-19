"""Tests for what this fork adds: hosted-model caching and token counting, the Ollama provider,
the LLM judge's agreement maths, and the cost table. All offline, no key, no model download."""
import json
from types import SimpleNamespace

from evals import final_table, judge
from triage.config import qwen_model, resolve_adapter
from triage.providers import OllamaProvider, OpenAIProvider

RECORD = '{"category": "ORDER", "intent": "cancel_order", "order_number": null}'


class FakeClient:
    def __init__(self, content: str, prompt_tokens=300, total_tokens=420):
        self.calls = 0
        self._content, self._usage = content, (prompt_tokens, total_tokens)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=self._usage[0], completion_tokens=20, total_tokens=self._usage[1]),
        )


def test_hosted_counts_thinking_tokens_and_caches(tmp_path):
    cache = tmp_path / "hosted.jsonl"
    client = FakeClient(RECORD)
    provider = OpenAIProvider("gemini-x", client=client, max_tokens=512, reasoning_effort="low", cache_path=cache)
    assert provider.complete(["cancel my order"]) == [RECORD]
    assert client.kwargs["reasoning_effort"] == "low" and client.kwargs["max_tokens"] == 512
    # Billed output = total - prompt, so hidden reasoning tokens are counted in the cost.
    assert (provider.input_tokens, provider.output_tokens) == (300, 120)
    again = OpenAIProvider("gemini-x", client=client, reasoning_effort="low", cache_path=cache)
    assert again.complete(["cancel my order"]) == [RECORD] and client.calls == 1
    assert again.output_tokens == 120


def test_ollama_provider_sends_raw_prompt():
    sent = []

    class FakeHttp:
        def post(self, path, json):
            sent.append((path, json))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": " " + RECORD})

    provider = OllamaProvider("triage-qwen3-0.6b")
    provider.http = FakeHttp()
    assert provider.complete(["cancel it"]) == [RECORD]
    path, body = sent[0]
    assert path == "/api/generate" and body["raw"] is True and body["prompt"].startswith("### Message\ncancel it")
    assert body["options"]["temperature"] == 0 and provider.name == "ollama-triage-qwen3-0.6b"
    assert len(provider.latencies_ms) == 1


def test_adapter_can_be_a_hub_repo_id(monkeypatch):
    assert resolve_adapter("someone/qwen3-0.6b-support-triage-lora") == "someone/qwen3-0.6b-support-triage-lora"
    assert str(resolve_adapter("adapter")).endswith("/adapter")
    monkeypatch.setenv("TRIAGE_ADAPTER_QWEN3_0_6B", "someone/x")
    assert qwen_model("qwen3-0.6b") == ("Qwen/Qwen3-0.6B", "someone/x")


def test_judge_agreement_maths():
    assert judge.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert judge.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    row = {"message": "cancel order ORD-12345", "output": RECORD.replace("null", '"ORD-12345"'),
           "expected": RECORD.replace("null", '"ORD-12345"')}
    assert judge.is_exact(row)
    assert not judge.is_exact({**row, "output": "not json"})


def test_judge_parses_grades_for_every_candidate():
    client = FakeClient(json.dumps({"grades": {"A": 5, "B": 2}, "notes": "B picks the wrong intent"}))
    out = judge.ask_judge(client, "judge-model", "msg", RECORD, {"A": RECORD, "B": "{}"}, "low")
    assert out["grades"] == {"A": 5, "B": 2}
    assert "Reference record" in client.kwargs["messages"][1]["content"]


def test_cost_per_thousand_requests():
    r = {"model": "gemini-3.8-flash", "n": 100, "input_tokens": 50_000, "output_tokens": 10_000}
    # 500 in + 100 out tokens per request: 1000 * (500 * 0.75 + 100 * 3.75) / 1e6 = $0.75
    assert final_table.cost_cell("hosted", r, 900.0, "api").startswith("$0.75 API")
    assert final_table.cost_cell("lora", {}, 1000.0, "cpu").startswith("$0 API")


def test_tables_are_written_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{final_table.START}\nold\n{final_table.END}\nend\n")
    assert final_table.write_into(readme, "NEW TABLE")
    assert readme.read_text() == f"intro\n{final_table.START}\nNEW TABLE\n{final_table.END}\nend\n"
