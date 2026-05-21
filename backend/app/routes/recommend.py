from fastapi import APIRouter, HTTPException
import os

from app.agents.advisor_agent import ENHANCED_KEY, SUMMARY_KEY
from app.evaluation.runner import evaluate_and_write
from app.agents.workflow import RecommendationState, run_recommendation_workflow
from app.guardrails.learning_guardrails import validate_learning_request, validate_recommendations
from app.learning.intelligence import (
    collaborative_peer_score,
    course_completion_rate,
    course_success_rate,
    popularity_percentiles,
    skill_graph_summary,
)
from app.llm.groq_client import GroqClient
from app.models.schemas import CourseRecommendation, RecommendRequest, RecommendResponse
from app.utils.tracing import trace_span
from app.utils.text import normalize_query


DESCRIPTION_MIN_WORDS = 100
DESCRIPTION_MAX_WORDS = 170


router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    return generate_recommendation(payload)


def generate_recommendation(payload: RecommendRequest) -> RecommendResponse:
    with trace_span(
        "recommend.generate",
        inputs={
            "query": payload.query,
            "student_level": payload.student_level,
            "career_goal": payload.career_goal,
            "top_k": payload.top_k,
        },
        metadata={"route": "/recommend"},
    ):
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
            preferred_skills=payload.preferred_skills,
            completed_courses=payload.completed_courses,
            liked_courses=payload.liked_courses,
            disliked_courses=payload.disliked_courses,
            learner_progress=payload.learner_progress,
            peer_group=payload.peer_group,
        )
        recommendation = recommendation_from_state(payload, state, guardrail_result.warnings)
        if os.getenv("SKIP_AUTO_EVALUATION", "").lower() not in {"1", "true", "yes"}:
            evaluate_and_write(recommendation)
        return recommendation


def recommendation_from_state(
    payload: RecommendRequest,
    state: RecommendationState,
    validation_warnings: list[str] | None = None,
) -> RecommendResponse:
    results = state.get("ranked_courses", [])
    prerequisite_gaps = state.get("prerequisite_gaps", {})
    explanations = state.get("explanations", {})
    popularity = popularity_percentiles()

    recommendations = [
        CourseRecommendation(
            title=_display_name(course.title),
            organization=_display_name(course.organization),
            difficulty=course.difficulty,
            rating=course.rating,
            url=course.url,
            skills=course.skills,
            description=_format_course_description(course.description),
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
            success_rate=course_success_rate(course),
            completion_rate=course_completion_rate(course),
            popularity_percentile=popularity.get(course.id, 0.0),
            matched_skills=_matched_skills(payload, course),
            peer_recommendation_score=round(
                collaborative_peer_score(
                    course,
                    current_skills=payload.current_skills,
                    preferred_skills=payload.preferred_skills,
                    career_goal=payload.career_goal,
                ),
                3,
            ),
        )
        for course, score in results
    ]

    return RecommendResponse(
        normalized_query=normalize_query(payload.query),
        recommendations=recommendations,
        learning_path=state.get("learning_path", []),
        career_alignment=state.get("career_alignment") or payload.career_goal,
        advisor_summary=explanations.get(SUMMARY_KEY),
        learner_domain=state.get("learner_domain"),
        agent_handoffs=state.get("agent_handoffs", []),
        agent_messages=state.get("agent_messages", []),
        skill_graph=skill_graph_summary(
            [
                *payload.current_skills,
                *payload.preferred_skills,
                *payload.skill_categories,
                *(skill for item in recommendations for skill in item.skills[:2]),
            ]
        ),
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


def _matched_skills(payload: RecommendRequest, course) -> list[str]:
    wanted = {
        *[skill.lower() for skill in payload.current_skills],
        *[skill.lower() for skill in payload.preferred_skills],
        *[skill.lower() for skill in payload.skill_categories],
    }
    query_terms = set(normalize_query(payload.query).split())
    matched = []
    for skill in course.skills:
        key = skill.lower()
        if key in wanted or any(term in key for term in query_terms):
            matched.append(skill)
    return matched[:8]


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


def _format_course_description(description: str) -> str:
    cleaned = _sentence_case(description)
    if _word_count(cleaned) <= DESCRIPTION_MAX_WORDS:
        return cleaned

    summarized = _summarize_description(cleaned)
    if summarized:
        return summarized

    return _truncate_words(cleaned, DESCRIPTION_MAX_WORDS)


def _summarize_description(description: str) -> str | None:
    client = GroqClient()
    if not client.enabled:
        return None

    with trace_span(
        "recommend.summarize_course_description",
        run_type="llm",
        inputs={"description_words": _word_count(description)},
    ):
        response = client.complete(
            system_prompt=(
                "You summarize course descriptions for a university course recommendation UI. "
                "Return only the rewritten description, with no heading, bullet points, markdown, "
                "or commentary. Keep the summary factual and student-facing."
            ),
            user_prompt=(
                f"Summarize this course description in {DESCRIPTION_MIN_WORDS}-{DESCRIPTION_MAX_WORDS} "
                "words. Preserve the main learning outcomes, topics, format, and audience. "
                "Do not add facts that are not present.\n\n"
                f"{description}"
            ),
        )
    if not response:
        return None

    summary = _sentence_case(response)
    words = _word_count(summary)
    if DESCRIPTION_MIN_WORDS <= words <= DESCRIPTION_MAX_WORDS:
        return summary
    if words > DESCRIPTION_MAX_WORDS:
        return _truncate_words(summary, DESCRIPTION_MAX_WORDS)
    return None


def _word_count(text: str) -> int:
    return len(text.split())


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words]).rstrip(" ,;:")
    if truncated and truncated[-1] not in ".!?":
        truncated = f"{truncated}..."
    return truncated
