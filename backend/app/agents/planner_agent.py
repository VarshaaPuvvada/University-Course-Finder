from app.graph.prerequisite_validator import build_learning_path, validate_sequence
from app.rag.course_repository import Course


def run_planner_agent(ranked_courses: list[tuple[Course, float]]) -> list[str]:
    path = build_learning_path([course for course, _ in ranked_courses])
    return validate_sequence(path)

