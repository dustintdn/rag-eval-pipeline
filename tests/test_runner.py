"""
Runner tests cover the settings-override context manager that powers CLI
A/B comparison runs without permanent .env edits.
"""
from config import settings
from eval.runner import _settings_override


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
