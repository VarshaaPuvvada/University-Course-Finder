from app.models.schemas import RecommendResponse


def evaluate_retrieval(response: RecommendResponse) -> dict[str, float]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import ResponseRelevancy, faithfulness
    except ImportError:
        return {
            "ragas_available": 0.0,
            "ragas_error": 1.0,
        }

    dataset = Dataset.from_dict(
        {
            "question": [response.normalized_query],
            "answer": [_recommendation_output(response)],
            "contexts": [[_recommendation_context(response)]],
        }
    )

    metrics: dict[str, float] = {"ragas_available": 1.0}
    try:
        result = evaluate(dataset, metrics=[faithfulness, ResponseRelevancy()])
        scores = _result_to_dict(result)
        for name, value in scores.items():
            metrics[f"ragas_{name}"] = _score(value)
    except Exception:
        metrics["ragas_runtime_error"] = 1.0

    total = max(len(response.recommendations), 1)
    with_explanations = sum(1 for item in response.recommendations if item.explanation)
    with_skills = sum(1 for item in response.recommendations if item.skills)
    metrics["retrieved_with_skills"] = round(with_skills / total, 3)
    metrics["retrieved_with_explanations"] = round(with_explanations / total, 3)
    return metrics


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


def _score(value) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0
