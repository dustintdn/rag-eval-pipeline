import json
from pathlib import Path

import pytest

from eval.dataset import EvalSample, load_dataset, save_dataset

SAMPLE: list[EvalSample] = [
    {
        "question": "What is RAG?",
        "ground_truth": "Retrieval-Augmented Generation",
        "contexts": ["chunk one", "chunk two"],
        "answer": "RAG is ...",
    },
    {
        "question": "What is Chroma?",
        "ground_truth": "A vector store",
        "contexts": ["chroma chunk"],
        "answer": "Chroma is ...",
    },
]


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "dataset.json"
    save_dataset(SAMPLE, path)
    loaded = load_dataset(path)
    assert loaded == SAMPLE


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "dataset.json"
    save_dataset(SAMPLE, path)
    assert path.exists()


def test_load_preserves_field_types(tmp_path):
    path = tmp_path / "dataset.json"
    save_dataset(SAMPLE, path)
    loaded = load_dataset(path)
    assert isinstance(loaded[0]["contexts"], list)
    assert isinstance(loaded[0]["question"], str)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nonexistent.json")


def test_save_writes_valid_json(tmp_path):
    path = tmp_path / "out.json"
    save_dataset(SAMPLE, path)
    raw = json.loads(path.read_text())
    assert isinstance(raw, list)
    assert len(raw) == len(SAMPLE)
