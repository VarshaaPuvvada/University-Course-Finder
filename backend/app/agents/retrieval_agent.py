from app.rag.course_repository import Course
from app.rag.retrieval_pipeline import retrieve_courses


def run_retrieval_agent(
    query: str,
    current_skills: list[str],
    student_level: str,
    top_k: int,
    organizations: list[str] | None = None,
    difficulties: list[str] | None = None,
    skill_categories: list[str] | None = None,
    min_rating: float | None = None,
    strict_difficulty: bool = False,
    preferred_skills: list[str] | None = None,
    completed_courses: list[str] | None = None,
    liked_courses: list[str] | None = None,
    disliked_courses: list[str] | None = None,
    learner_progress: float | None = None,
    career_goal: str | None = None,
) -> list[tuple[Course, float]]:
    return retrieve_courses(
        query=query,
        current_skills=current_skills,
        student_level=student_level,
        top_k=top_k,
        organizations=organizations,
        difficulties=difficulties,
        skill_categories=skill_categories,
        min_rating=min_rating,
        strict_difficulty=strict_difficulty,
        preferred_skills=preferred_skills,
        completed_courses=completed_courses,
        liked_courses=liked_courses,
        disliked_courses=disliked_courses,
        learner_progress=learner_progress,
        career_goal=career_goal,
    )
