from dataclasses import dataclass

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from retriever.retriever import get_retriever

_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
For every claim you make, cite the source file in parentheses, e.g. (source: report.pdf).
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


@dataclass
class QAResult:
    answer: str
    source_documents: list[Document]


def build_chain(top_k: int | None = None) -> RetrievalQA:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=get_retriever(top_k),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def ask(question: str, top_k: int | None = None) -> QAResult:
    chain = build_chain(top_k)
    result = chain.invoke({"query": question})
    return QAResult(
        answer=result["result"],
        source_documents=result["source_documents"],
    )
