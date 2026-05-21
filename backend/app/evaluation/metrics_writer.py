from datetime import UTC, datetime
from pathlib import Path

from app.models.schemas import RecommendResponse


METRICS_PATH = Path(__file__).resolve().parents[2] / "evaluation_metrics.txt"


def write_evaluation_metrics(
    recommendation: RecommendResponse,
    retrieval_metrics: dict[str, float],
    agent_metrics: dict[str, float],
) -> Path:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "Intelligent University Course Finder - Evaluation Metrics",
        f"Generated at: {timestamp}",
        "",
        f"Normalized query: {recommendation.normalized_query}",
        f"Recommendation count: {len(recommendation.recommendations)}",
        f"Learning path length: {len(recommendation.learning_path)}",
        f"Career alignment: {recommendation.career_alignment or 'N/A'}",
        "",
        "Retrieval metrics:",
    ]
    lines.extend(
        f"- {name}: {value:.3f}"
        for name, value in sorted(retrieval_metrics.items())
    )
    lines.append("")
    lines.append("Agent metrics:")
    lines.extend(
        f"- {name}: {value:.3f}"
        for name, value in sorted(agent_metrics.items())
    )
    lines.append("")
    lines.append("Recommendations:")
    for index, item in enumerate(recommendation.recommendations, start=1):
        lines.extend(
            [
                f"{index}. {item.title}",
                f"   Organization: {item.organization}",
                f"   Difficulty: {item.difficulty}",
                f"   Final score: {item.final_score:.4f}",
                f"   Skills: {', '.join(item.skills[:8]) or 'N/A'}",
                f"   Prerequisite gaps: {', '.join(item.prerequisite_gaps) or 'None'}",
            ]
        )
    lines.append("")

    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return METRICS_PATH
