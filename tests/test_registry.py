import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prompts.registry import DEFAULT_VERSION, list_versions, load_prompt


def test_load_default_prompt_returns_template_and_meta():
    prompt, meta = load_prompt(DEFAULT_VERSION)
    assert "{context}" in prompt.template
    assert "{question}" in prompt.template
    assert meta["version"] == DEFAULT_VERSION
    assert "description" in meta


def test_load_all_shipped_versions():
    for version in list_versions():
        prompt, meta = load_prompt(version)
        assert "{context}" in prompt.template
        assert "{question}" in prompt.template
        assert meta["version"] == version


def test_load_unknown_version_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt("v99_does_not_exist")


def test_list_versions_is_sorted():
    versions = list_versions()
    assert versions == sorted(versions)


def test_list_versions_contains_shipped_prompts():
    versions = list_versions()
    assert "v1_cite_sources" in versions
    assert "v2_concise" in versions


def test_load_prompt_from_custom_file(tmp_path):
    custom = {
        "version": "v_test",
        "description": "test prompt",
        "template": "Context: {context}\nQ: {question}\nA:",
    }
    prompt_file = tmp_path / "v_test.json"
    prompt_file.write_text(json.dumps(custom))

    with patch("prompts.registry._PROMPTS_DIR", tmp_path):
        prompt, meta = load_prompt("v_test")

    assert meta["version"] == "v_test"
    assert "{context}" in prompt.template


def test_prompt_input_variables():
    prompt, _ = load_prompt(DEFAULT_VERSION)
    assert set(prompt.input_variables) == {"context", "question"}
