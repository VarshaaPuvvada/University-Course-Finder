from dataclasses import dataclass, field

from app.llm.groq_client import GroqClient
from app.rag.course_repository import Course
from app.utils.text import normalize_query


BLOCKED_TERMS = {
    "cheat",
    "exam answers",
    "plagiarize",
    "plagiarism",
    "bypass proctor",
    "hack account",
    "steal credentials",
}

LEARNING_INTENT_TERMS = {
    "course",
    "learn",
    "study",
    "understand",
    "skill",
    "career",
    "degree",
    "class",
    "training",
    "certificate",
    "programming",
    "data",
    "finance",
    "machine",
    "cybersecurity",
    "software",
    "business",
}


@dataclass
class GuardrailResult:
    allowed: bool
    warnings: list[str] = field(default_factory=list)


def validate_learning_request(query: str, career_goal: str | None = None) -> GuardrailResult:
    normalized = normalize_query(" ".join(part for part in [query, career_goal or ""] if part))
    warnings: list[str] = []

    if len(normalized.split()) < 2:
        return GuardrailResult(False, ["Please provide a clearer learning goal."])

    blocked = [term for term in BLOCKED_TERMS if term in normalized]
    if blocked:
        return GuardrailResult(
            False,
            ["The request appears to involve academic dishonesty or unsafe activity."],
        )

    if not any(term in normalized for term in LEARNING_INTENT_TERMS):
        warnings.append("The query is broad; recommendations may be less precise.")

    return GuardrailResult(True, warnings)


def validate_recommendations(
    query: str,
    courses: list[Course],
    current_skills: list[str],
    use_llm_judge: bool = False,
) -> list[str]:
    warnings: list[str] = []
    known = {skill.lower() for skill in current_skills}

    advanced_without_background = [
        course.title
        for course in courses
        if course.difficulty == "advanced" and not known
    ]
    if advanced_without_background:
        warnings.append(
            "Advanced courses were found for a learner with no listed background skills."
        )

    if use_llm_judge:
        judge_warning = _llm_judge_warning(query, courses)
        if judge_warning:
            warnings.append(judge_warning)

    return warnings


def _llm_judge_warning(query: str, courses: list[Course]) -> str | None:
    client = GroqClient()
    if not client.enabled or not courses:
        return None

    course_brief = "\n".join(
        f"- {course.title}: {', '.join(course.skills[:6])}; difficulty={course.difficulty}"
        for course in courses[:5]
    )
    response = client.complete(
        system_prompt=(
            "You are an academic advising judge. Reply with one short warning only if "
            "the recommendations are unsafe, irrelevant, or have prerequisite issues. "
            "Reply exactly OK if they are acceptable."
        ),
        user_prompt=f"Student query: {query}\nRecommended courses:\n{course_brief}",
    )
    if not response:
        return None

    cleaned = response.strip()
    if cleaned.upper() == "OK":
        return None
    return f"LLM judge warning: {cleaned}"
