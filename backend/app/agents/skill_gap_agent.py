from app.graph.prerequisite_validator import find_prerequisite_gaps
from app.rag.course_repository import Course


def run_skill_gap_agent(
    ranked_courses: list[tuple[Course, float]],
    current_skills: list[str],
) -> dict[str, list[str]]:
    return {
        course.id: find_prerequisite_gaps(course, current_skills)
        for course, _ in ranked_courses
    }

