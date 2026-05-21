import tempfile
from pathlib import Path

import pytest

from ingest.loader import load_directory, load_file


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_load_txt_file(tmp_path):
    p = _write(tmp_path, "doc.txt", "Hello world")
    docs = load_file(p)
    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content


def test_load_md_file(tmp_path):
    p = _write(tmp_path, "doc.md", "# Title\nContent here")
    docs = load_file(p)
    assert len(docs) >= 1
    combined = " ".join(d.page_content for d in docs)
    assert "Content here" in combined


def test_load_file_attaches_source_file_metadata(tmp_path):
    p = _write(tmp_path, "notes.txt", "some text")
    docs = load_file(p)
    assert all(d.metadata.get("source_file") == "notes.txt" for d in docs)


def test_load_unsupported_type_raises(tmp_path):
    p = _write(tmp_path, "data.csv", "a,b,c")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(p)


def test_load_directory_finds_all_supported_types(tmp_path):
    _write(tmp_path, "a.txt", "text file")
    _write(tmp_path, "b.md", "markdown file")
    # .csv should be ignored
    _write(tmp_path, "c.csv", "should be ignored")
    docs = load_directory(tmp_path)
    assert len(docs) == 2


def test_load_directory_recurses_into_subdirs(tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    _write(tmp_path, "root.txt", "root doc")
    _write(subdir, "nested.txt", "nested doc")
    docs = load_directory(tmp_path)
    assert len(docs) == 2


def test_load_directory_empty(tmp_path):
    docs = load_directory(tmp_path)
    assert docs == []
