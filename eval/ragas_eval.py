from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import settings
from eval.dataset import EvalSample

_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _llm():
    return LangchainLLMWrapper(
        ChatOpenAI(model=settings.llm_model, openai_api_key=settings.openai_api_key, temperature=0)
    )


def _embeddings():
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
    )


def run_ragas(samples: list[EvalSample]) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return (mean_scores, per_sample_scores). per_sample[i][metric] = score for sample i."""
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["ground_truth"],
            )
            for s in samples
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=_METRICS,
        llm=_llm(),
        embeddings=_embeddings(),
    )
    means = {m.name: sum(result[m.name]) / len(result[m.name]) for m in _METRICS}
    per_sample = [
        {m.name: float(result[m.name][i]) for m in _METRICS}
        for i in range(len(samples))
    ]
    return means, per_sample
