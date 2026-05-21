from app.models.schemas import RecommendResponse


def evaluate_agents(response: RecommendResponse) -> dict[str, float]:
    try:
        from app.evaluation.groq_deepeval_model import GroqDeepEvalModel
        from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        return {
            "deepeval_available": 0.0,
            "deepeval_error": 1.0,
        }

    context = _recommendation_context(response)
    actual_output = _recommendation_output(response)
    test_case = LLMTestCase(
        input=response.normalized_query,
        actual_output=actual_output,
        context=[context],
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
        answer_relevancy.measure(test_case)
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
        hallucination.measure(test_case)
        metrics["deepeval_hallucination"] = _score(hallucination.score)
        metrics["deepeval_hallucination_success"] = float(bool(hallucination.success))
    except Exception:
        metrics["deepeval_hallucination_error"] = 1.0

    total = max(len(response.recommendations), 1)
    gap_coverage = sum(1 for item in response.recommendations if item.prerequisite_gaps is not None)
    metrics["skill_gap_coverage"] = round(gap_coverage / total, 3)
    return metrics


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
