from app.models.schemas import RecommendResponse


def evaluate_retrieval(response: RecommendResponse) -> dict[str, float]:
    total = max(len(response.recommendations), 1)
    with_explanations = sum(1 for item in response.recommendations if item.explanation)
    with_skills = sum(1 for item in response.recommendations if item.skills)
    return {
        "context_precision": round(with_skills / total, 3),
        "faithfulness": round(with_explanations / total, 3),
        "recall_proxy": round(min(len(response.learning_path) / 5, 1.0), 3),
    }

