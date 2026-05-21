from fastapi import APIRouter

from app.agents.workflow import RecommendationState, run_recommendation_workflow
from app.models.schemas import CourseRecommendation, RecommendRequest, RecommendResponse
from app.utils.text import normalize_query


router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    return generate_recommendation(payload)


def generate_recommendation(payload: RecommendRequest) -> RecommendResponse:
    state = run_recommendation_workflow(
        query=payload.query,
        current_skills=payload.current_skills,
        student_level=payload.student_level,
        career_goal=payload.career_goal,
        top_k=payload.top_k,
    )
    return recommendation_from_state(payload, state)


def recommendation_from_state(
    payload: RecommendRequest,
    state: RecommendationState,
) -> RecommendResponse:
    results = state.get("ranked_courses", [])
    prerequisite_gaps = state.get("prerequisite_gaps", {})
    explanations = state.get("explanations", {})

    recommendations = [
        CourseRecommendation(
            title=course.title,
            organization=course.organization,
            difficulty=course.difficulty,
            rating=course.rating,
            url=course.url,
            skills=course.skills,
            explanation=explanations.get(
                course.id,
                (
                    f"Matches your goal through {', '.join(course.skills[:4]) or 'the course description'} "
                    f"and is rated {course.rating or 'N/A'}."
                ),
            ),
            prerequisite_gaps=prerequisite_gaps.get(course.id, []),
            final_score=round(score, 4),
        )
        for course, score in results
    ]

    return RecommendResponse(
        normalized_query=normalize_query(payload.query),
        recommendations=recommendations,
        learning_path=state.get("learning_path", []),
        career_alignment=state.get("career_alignment") or payload.career_goal,
    )
