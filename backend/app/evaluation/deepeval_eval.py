import os

from app.models.schemas import RecommendResponse
from app.utils.env import load_backend_env
from app.utils.tracing import trace_span


def evaluate_agents(response: RecommendResponse) -> dict[str, float]:
    with trace_span(
        "evaluation.deepeval",
        inputs={
            "normalized_query": response.normalized_query,
            "recommendation_count": len(response.recommendations),
        },
        metadata={"framework": "deepeval"},
    ):
        load_backend_env()
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
        try:
            from app.evaluation.groq_deepeval_model import GroqDeepEvalModel
            from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
            from deepeval.test_case import LLMTestCase
        except Exception:
            return {
                "deepeval_available": 0.0,
                "deepeval_error": 1.0,
                **_structural_agent_metrics(response),
            }

        if not os.getenv("GROQ_API_KEY"):
            return {
                "deepeval_available": 1.0,
                "deepeval_skipped_missing_groq_key": 1.0,
                **_structural_agent_metrics(response),
            }

        context = _recommendation_context(response)
        actual_output = _recommendation_output(response)
        test_case = LLMTestCase(
            input=response.normalized_query,
            actual_output=actual_output,
            context=[context],
            retrieval_context=[context],
        )

        metrics: dict[str, float] = {"deepeval_available": 1.0}
        evaluator_model = GroqDeepEvalModel()
        try:
            answer_relevancy = AnswerRelevancyMetric(
                threshold=0.5,
                include_reason=True,
                async_mode=False,
                model=evaluator_model,
            )
            answer_relevancy.measure(
                test_case,
                _show_indicator=False,
                _log_metric_to_confident=False,
            )
            metrics["deepeval_answer_relevancy"] = _score(answer_relevancy.score)
            metrics["deepeval_answer_relevancy_success"] = float(bool(answer_relevancy.success))
        except Exception:
            metrics["deepeval_answer_relevancy_error"] = 1.0

        try:
            hallucination = HallucinationMetric(
                threshold=0.5,
                include_reason=True,
                async_mode=False,
                model=evaluator_model,
            )
            hallucination.measure(
                test_case,
                _show_indicator=False,
                _log_metric_to_confident=False,
            )
            metrics["deepeval_hallucination"] = _score(hallucination.score)
            metrics["deepeval_hallucination_success"] = float(bool(hallucination.success))
        except Exception:
            metrics["deepeval_hallucination_error"] = 1.0

        metrics.update(_structural_agent_metrics(response))
        return metrics


def _structural_agent_metrics(response: RecommendResponse) -> dict[str, float]:
    total = max(len(response.recommendations), 1)
    gap_coverage = sum(1 for item in response.recommendations if item.prerequisite_gaps is not None)
    return {"skill_gap_coverage": round(gap_coverage / total, 3)}


def _recommendation_context(response: RecommendResponse) -> str:
    parts: list[str] = []
    for item in response.recommendations:
        parts.append(
            " | ".join(
                [
                    f"title={item.title}",
                    f"organization={item.organization}",
                    f"difficulty={item.difficulty}",
                    f"rating={item.rating}",
                    f"skills={', '.join(item.skills)}",
                    f"description={item.description}",
                ]
            )
        )
    return "\n".join(parts)


def _recommendation_output(response: RecommendResponse) -> str:
    return "\n".join(
        f"{item.title}: {item.explanation}" for item in response.recommendations
    )


def _score(value) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0
