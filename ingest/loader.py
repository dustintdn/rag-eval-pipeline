from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,  # UnstructuredMarkdownLoader requires the heavy `unstructured` package
}


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    loader_cls = _LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    docs = loader_cls(str(path)).load()
    for doc in docs:
        doc.metadata.setdefault("source_file", path.name)
    return docs


def load_directory(source_dir: str | Path) -> list[Document]:
    source_dir = Path(source_dir)
    docs: list[Document] = []
    for ext in _LOADERS:
        for file in sorted(source_dir.glob(f"**/*{ext}")):
            docs.extend(load_file(file))
    return docs
