import os
import math
import asyncio

from app.models.schemas import RecommendResponse
from app.rag.embedding_service import OpenRouterEmbeddingService
from app.utils.env import load_backend_env
from app.utils.tracing import trace_span


def evaluate_retrieval(response: RecommendResponse) -> dict[str, float]:
    with trace_span(
        "evaluation.ragas",
        inputs={
            "normalized_query": response.normalized_query,
            "recommendation_count": len(response.recommendations),
        },
        metadata={"framework": "ragas"},
    ):
        load_backend_env()
        try:
            os.environ.setdefault("USE_TORCH", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
            from datasets import Dataset
            from langchain_openai import ChatOpenAI
            from ragas import evaluate
            from ragas.embeddings import BaseRagasEmbeddings
            from ragas.metrics import ResponseRelevancy, faithfulness
        except Exception:
            return {
                "ragas_available": 0.0,
                "ragas_error": 1.0,
                **_structural_retrieval_metrics(response),
            }

        if not os.getenv("GROQ_API_KEY"):
            return {
                "ragas_available": 1.0,
                "ragas_skipped_missing_groq_key": 1.0,
                **_structural_retrieval_metrics(response),
            }

        if not os.getenv("OPENROUTER_API_KEY"):
            return {
                "ragas_available": 1.0,
                "ragas_skipped_missing_openrouter_key": 1.0,
                **_structural_retrieval_metrics(response),
            }

        dataset = Dataset.from_dict(
            {
                "user_input": [response.normalized_query],
                "response": [_recommendation_output(response)],
                "retrieved_contexts": [[_recommendation_context(response)]],
            }
        )

        metrics: dict[str, float] = {"ragas_available": 1.0}
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, ResponseRelevancy(strictness=1)],
                llm=_ragas_llm(ChatOpenAI),
                embeddings=_ragas_embeddings(BaseRagasEmbeddings),
                show_progress=False,
                raise_exceptions=False,
            )
            scores = _result_to_dict(result)
            for name, value in scores.items():
                score = _score(value)
                if score is not None:
                    metrics[f"ragas_{name}"] = score
        except Exception:
            metrics["ragas_runtime_error"] = 1.0

        metrics.update(_structural_retrieval_metrics(response))
        return metrics


def _ragas_llm(chat_openai_cls):
    return chat_openai_cls(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model=os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )


def _ragas_embeddings(base_embeddings_cls):
    class ProjectOpenRouterEmbeddings(base_embeddings_cls):
        def __init__(self) -> None:
            super().__init__()
            self.service = OpenRouterEmbeddingService()

        def embed_query(self, text: str) -> list[float]:
            embedding = self.service.embed_text(text)
            if not embedding:
                raise ValueError("OpenRouter returned no embedding for the query.")
            return embedding

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embeddings = self.service.embed_texts(texts)
            if len(embeddings) != len(texts):
                raise ValueError("OpenRouter returned an unexpected number of embeddings.")
            return embeddings

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    return ProjectOpenRouterEmbeddings()


def _recommendation_context(response: RecommendResponse) -> str:
    return "\n".join(
        f"{item.title}. Skills: {', '.join(item.skills)}. Description: {item.description}"
        for item in response.recommendations
    )


def _recommendation_output(response: RecommendResponse) -> str:
    return "\n".join(
        f"{item.title}: {item.explanation}" for item in response.recommendations
    )


def _result_to_dict(result) -> dict:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if not frame.empty:
            return frame.iloc[0].to_dict()
    if hasattr(result, "scores") and result.scores:
        return result.scores[0]
    if isinstance(result, dict):
        return result
    return {}


def _structural_retrieval_metrics(response: RecommendResponse) -> dict[str, float]:
    total = max(len(response.recommendations), 1)
    with_explanations = sum(1 for item in response.recommendations if item.explanation)
    with_skills = sum(1 for item in response.recommendations if item.skills)
    return {
        "retrieved_with_skills": round(with_skills / total, 3),
        "retrieved_with_explanations": round(with_explanations / total, 3),
    }


def _score(value) -> float | None:
    if isinstance(value, str):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return round(score, 3)
