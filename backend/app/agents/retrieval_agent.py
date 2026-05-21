from app.rag.course_repository import Course
from app.rag.retrieval_pipeline import retrieve_courses


def run_retrieval_agent(
    query: str,
    current_skills: list[str],
    student_level: str,
    top_k: int,
) -> list[tuple[Course, float]]:
    return retrieve_courses(
        query=query,
        current_skills=current_skills,
        student_level=student_level,
        top_k=top_k,
    )

