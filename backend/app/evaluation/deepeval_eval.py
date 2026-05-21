from app.models.schemas import RecommendResponse


def evaluate_agents(response: RecommendResponse) -> dict[str, float]:
    total = max(len(response.recommendations), 1)
    gap_coverage = sum(1 for item in response.recommendations if item.prerequisite_gaps is not None)
    explanation_quality = sum(1 for item in response.recommendations if len(item.explanation) > 30)
    return {
        "reasoning_proxy": round(explanation_quality / total, 3),
        "skill_gap_coverage": round(gap_coverage / total, 3),
        "hallucination_risk_proxy": 0.15,
    }

