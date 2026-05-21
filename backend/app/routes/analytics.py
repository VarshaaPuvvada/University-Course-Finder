from collections import Counter

from fastapi import APIRouter

from app.models.schemas import AnalyticsResponse
from app.rag.course_repository import load_courses


router = APIRouter(tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics() -> AnalyticsResponse:
    courses = load_courses()
    skill_counts = Counter(skill for course in courses for skill in course.skills)
    difficulty_counts = Counter(course.difficulty for course in courses)
    organization_counts = Counter(course.organization for course in courses)
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
    )

