import json
from pathlib import Path

import pytest

from eval.dataset import EvalSample, dataset_checksum, load_dataset, save_dataset

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


def test_dataset_checksum_is_stable():
    assert dataset_checksum(SAMPLE) == dataset_checksum(SAMPLE)


def test_dataset_checksum_changes_on_edit():
    edited: list[EvalSample] = [
        {**SAMPLE[0], "ground_truth": "Different ground truth"},
        SAMPLE[1],
    ]
    assert dataset_checksum(edited) != dataset_checksum(SAMPLE)


def test_dataset_checksum_field_order_independent():
    """Reordering keys in a dict shouldn't change the checksum."""
    reordered = [
        {"answer": s["answer"], "contexts": s["contexts"],
         "ground_truth": s["ground_truth"], "question": s["question"]}
        for s in SAMPLE
    ]
    assert dataset_checksum(reordered) == dataset_checksum(SAMPLE)
