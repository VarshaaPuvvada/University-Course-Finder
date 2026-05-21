from fastapi import APIRouter, HTTPException

from app.agents.advisor_agent import ENHANCED_KEY, SUMMARY_KEY
from app.agents.workflow import RecommendationState, run_recommendation_workflow
from app.guardrails.learning_guardrails import validate_learning_request, validate_recommendations
from app.models.schemas import CourseRecommendation, RecommendRequest, RecommendResponse
from app.utils.text import normalize_query


router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    return generate_recommendation(payload)


def generate_recommendation(payload: RecommendRequest) -> RecommendResponse:
    guardrail_result = validate_learning_request(payload.query, payload.career_goal)
    if not guardrail_result.allowed:
        raise HTTPException(status_code=400, detail=guardrail_result.warnings)

    state = run_recommendation_workflow(
        query=payload.query,
        current_skills=payload.current_skills,
        student_level=payload.student_level,
        career_goal=payload.career_goal,
        top_k=payload.top_k,
        organizations=payload.organizations,
        difficulties=payload.difficulties,
        skill_categories=payload.skill_categories,
        min_rating=payload.min_rating,
        strict_difficulty=payload.strict_difficulty,
    )
    return recommendation_from_state(payload, state, guardrail_result.warnings)


def recommendation_from_state(
    payload: RecommendRequest,
    state: RecommendationState,
    validation_warnings: list[str] | None = None,
) -> RecommendResponse:
    results = state.get("ranked_courses", [])
    prerequisite_gaps = state.get("prerequisite_gaps", {})
    explanations = state.get("explanations", {})

    recommendations = [
        CourseRecommendation(
            title=_display_name(course.title),
            organization=_display_name(course.organization),
            difficulty=course.difficulty,
            rating=course.rating,
            url=course.url,
            skills=course.skills,
            description=_sentence_case(course.description),
            course_type=_display_name(course.course_type or "course"),
            duration=course.duration,
            review_count=course.review_count,
            explanation=explanations.get(
                course.id,
                (
                    f"Matches your goal through {', '.join(course.skills[:4]) or 'the course description'} "
                    f"and is rated {course.rating or 'N/A'}."
                ),
            ),
            prerequisite_gaps=prerequisite_gaps.get(course.id, []),
            final_score=round(score, 4),
            llm_enhanced=explanations.get(ENHANCED_KEY) == "true",
        )
        for course, score in results
    ]

    return RecommendResponse(
        normalized_query=normalize_query(payload.query),
        recommendations=recommendations,
        learning_path=state.get("learning_path", []),
        career_alignment=state.get("career_alignment") or payload.career_goal,
        advisor_summary=explanations.get(SUMMARY_KEY),
        validation_warnings=[
            *(validation_warnings or []),
            *validate_recommendations(
                query=payload.query,
                courses=[course for course, _ in results],
                current_skills=payload.current_skills,
                use_llm_judge=payload.use_llm_judge,
            ),
        ],
    )


def _display_name(value: str) -> str:
    words = {
        "ai": "AI",
        "api": "API",
        "aws": "AWS",
        "c++": "C++",
        "css": "CSS",
        "gcp": "GCP",
        "html": "HTML",
        "ibm": "IBM",
        "it": "IT",
        "sql": "SQL",
        "ui": "UI",
        "ux": "UX",
    }
    return " ".join(words.get(part.lower(), part.capitalize()) for part in value.split())


def _sentence_case(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    result: list[str] = []
    capitalize_next = True
    for char in cleaned:
        if capitalize_next and char.isalpha():
            result.append(char.upper())
            capitalize_next = False
        else:
            result.append(char)
        if char in ".!?":
            capitalize_next = True
    return "".join(result)
