from collections import Counter

from fastapi import APIRouter

from app.models.schemas import AnalyticsResponse
from app.learning.intelligence import (
    course_completion_rate,
    course_success_rate,
    skill_graph_summary,
)
from app.rag.course_repository import load_courses


router = APIRouter(tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics() -> AnalyticsResponse:
    courses = load_courses()
    skill_counts = Counter(skill for course in courses for skill in course.skills)
    difficulty_counts = Counter(course.difficulty for course in courses)
    organization_counts = Counter(course.organization for course in courses)
    success_by_difficulty = {}
    completion_by_difficulty = {}
    for difficulty in difficulty_counts:
        matching = [course for course in courses if course.difficulty == difficulty]
        success_by_difficulty[difficulty] = round(
            sum(course_success_rate(course) for course in matching) / max(len(matching), 1),
            3,
        )
        completion_by_difficulty[difficulty] = round(
            sum(course_completion_rate(course) for course in matching) / max(len(matching), 1),
            3,
        )
    high_success = sorted(
        courses,
        key=lambda course: (course_success_rate(course), course.review_count),
        reverse=True,
    )[:8]
    return AnalyticsResponse(
        total_courses=len(courses),
        popular_skills=[
            {"skill": skill, "count": count}
            for skill, count in skill_counts.most_common(10)
        ],
        difficulty_distribution=dict(difficulty_counts),
        top_organizations=[
            {"organization": org, "count": count}
            for org, count in organization_counts.most_common(10)
        ],
        success_rate_by_difficulty=success_by_difficulty,
        completion_rate_by_difficulty=completion_by_difficulty,
        high_success_courses=[
            {
                "title": course.title,
                "organization": course.organization,
                "success_rate": course_success_rate(course),
                "completion_rate": course_completion_rate(course),
                "rating": course.rating or 0.0,
                "review_count": course.review_count,
            }
            for course in high_success
        ],
        skill_graph_overview=skill_graph_summary([], limit=10),
    )
