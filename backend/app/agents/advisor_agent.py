from app.rag.course_repository import Course


def run_advisor_agent(
    query: str,
    ranked_courses: list[tuple[Course, float]],
    prerequisite_gaps: dict[str, list[str]],
) -> dict[str, str]:
    explanations: dict[str, str] = {}
    for course, _ in ranked_courses:
        skills = ", ".join(course.skills[:4]) or "the course topics"
        gaps = prerequisite_gaps.get(course.id, [])
        gap_text = f" Review {', '.join(gaps)} first." if gaps else " It fits your current readiness."
        explanations[course.id] = (
            f"Recommended for '{query}' because it covers {skills} at {course.difficulty} level."
            f"{gap_text}"
        )
    return explanations

