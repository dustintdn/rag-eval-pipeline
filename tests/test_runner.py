"""
Runner tests cover the settings-override context manager that powers CLI
A/B comparison runs without permanent .env edits, plus latency tracking.
"""
from unittest.mock import patch

from config import settings
from eval.runner import _settings_override, generate_live_samples


def test_settings_override_applies_and_restores():
    original_top_k = settings.top_k
    original_prompt = settings.prompt_version

    with _settings_override({"top_k": 99, "prompt_version": "v2_concise"}):
        assert settings.top_k == 99
        assert settings.prompt_version == "v2_concise"

    assert settings.top_k == original_top_k
    assert settings.prompt_version == original_prompt


def test_settings_override_none_is_noop():
    original_top_k = settings.top_k
    with _settings_override(None):
        assert settings.top_k == original_top_k
    assert settings.top_k == original_top_k


def test_settings_override_restores_on_exception():
    original_top_k = settings.top_k
    try:
        with _settings_override({"top_k": 42}):
            assert settings.top_k == 42
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert settings.top_k == original_top_k


def test_settings_override_handles_reranker_flag():
    original = settings.enable_reranker
    with _settings_override({"enable_reranker": not original}):
        assert settings.enable_reranker is (not original)
    assert settings.enable_reranker is original


def test_generate_live_samples_records_latency_and_tokens():
    from langchain_core.documents import Document
    from chain.qa_chain import QAResult

    fake = QAResult(
        answer="ans",
        source_documents=[Document(page_content="ctx", metadata={})],
        prompt_version="v1",
        token_usage={"prompt": 10, "completion": 5, "total": 15},
    )
    samples = [
        {"question": "q1", "ground_truth": "g1", "contexts": [], "answer": ""},
        {"question": "q2", "ground_truth": "g2", "contexts": [], "answer": ""},
    ]
    with patch("chain.qa_chain.ask", return_value=fake):
        out, latencies, tokens = generate_live_samples(samples)

    assert len(out) == 2
    assert len(latencies) == 2
    assert all(l >= 0 for l in latencies)
    assert out[0]["answer"] == "ans"
    assert tokens == [{"prompt": 10, "completion": 5, "total": 15}] * 2


def test_generate_live_samples_handles_missing_token_usage():
    from langchain_core.documents import Document
    from chain.qa_chain import QAResult

    fake = QAResult(
        answer="ans",
        source_documents=[Document(page_content="ctx", metadata={})],
        prompt_version="v1",
        token_usage=None,
    )
    with patch("chain.qa_chain.ask", return_value=fake):
        _, _, tokens = generate_live_samples([{"question": "q", "ground_truth": "g", "contexts": [], "answer": ""}])

    assert tokens == [None]
