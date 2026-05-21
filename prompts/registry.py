import json
from pathlib import Path

from langchain_core.prompts import PromptTemplate

_PROMPTS_DIR = Path(__file__).parent
DEFAULT_VERSION = "v1_cite_sources"


def load_prompt(version: str = DEFAULT_VERSION) -> tuple[PromptTemplate, dict]:
    """Return (PromptTemplate, metadata_dict) for the given version ID."""
    path = _PROMPTS_DIR / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version '{version}' not found in {_PROMPTS_DIR}")
    data = json.loads(path.read_text())
    prompt = PromptTemplate(
        template=data["template"],
        input_variables=["context", "question"],
    )
    meta = {"version": data["version"], "description": data["description"]}
    return prompt, meta


def list_versions() -> list[str]:
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.json"))
