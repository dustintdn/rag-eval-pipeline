import json
from pathlib import Path
from typing import TypedDict


class EvalSample(TypedDict):
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str


def load_dataset(path: str | Path) -> list[EvalSample]:
    with open(path) as f:
        return json.load(f)


def save_dataset(samples: list[EvalSample], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
