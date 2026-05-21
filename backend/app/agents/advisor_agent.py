import json

from app.learning.intelligence import token_budget_text
from app.llm.groq_client import GroqClient
from app.rag.course_repository import Course


SUMMARY_KEY = "__advisor_summary__"
ENHANCED_KEY = "__llm_enhanced__"


def run_advisor_agent(
    query: str,
    ranked_courses: list[tuple[Course, float]],
    prerequisite_gaps: dict[str, list[str]],
) -> dict[str, str]:
    llm_output = _llm_enhanced_advice(query, ranked_courses, prerequisite_gaps)
    if llm_output:
        print("[llm-advisor] Enhanced recommendation output generated.")
        return llm_output

    explanations: dict[str, str] = {}
    for course, _ in ranked_courses:
        skills = ", ".join(course.skills[:4]) or "the course topics"
        gaps = prerequisite_gaps.get(course.id, [])
        gap_text = f" Review {', '.join(gaps)} first." if gaps else " It fits your current readiness."
        explanations[course.id] = (
            f"{_display_name(course.title)} is recommended because it covers {skills} "
            f"at {course.difficulty} level for your goal: '{query}'."
            f"{gap_text}"
        )
    explanations[SUMMARY_KEY] = (
        "Recommendations were prepared from the retrieved course metadata, skill matches, "
        "difficulty fit, and prerequisite checks."
    )
    explanations[ENHANCED_KEY] = "false"
    return explanations


def _llm_enhanced_advice(
    query: str,
    ranked_courses: list[tuple[Course, float]],
    prerequisite_gaps: dict[str, list[str]],
) -> dict[str, str] | None:
    client = GroqClient()
    if not client.enabled or not ranked_courses:
        return None

    course_payload = [
        {
            "id": course.id,
            "title": _display_name(course.title),
            "organization": _display_name(course.organization),
            "difficulty": course.difficulty,
            "rating": course.rating,
            "duration": course.duration,
            "type": course.course_type,
            "skills": course.skills[:8],
            "prerequisite_gaps": prerequisite_gaps.get(course.id, []),
            "description": token_budget_text(course.description, 120),
            "score": round(score, 4),
        }
        for course, score in ranked_courses
    ]
    response = client.complete(
        system_prompt=(
            "You are a concise academic course advisor. Improve the recommendation text "
            "so it is polished, specific, and useful for a student. Use only the provided "
            "course metadata. Return strict JSON with keys: summary and explanations. "
            "explanations must map course id strings to one paragraph each."
        ),
        user_prompt=json.dumps(
            {
                "student_query": query,
                "retrieved_courses": course_payload,
            },
            ensure_ascii=True,
        ),
    )
    if not response:
        return None

    try:
        parsed = json.loads(_extract_json(response))
    except json.JSONDecodeError:
        return None

    explanations = parsed.get("explanations")
    if not isinstance(explanations, dict):
        return None

    output = {
        str(course_id): str(text).strip()
        for course_id, text in explanations.items()
        if str(text).strip()
    }
    if not output:
        return None

    output[SUMMARY_KEY] = str(parsed.get("summary") or "").strip()
    output[ENHANCED_KEY] = "true"
    return output


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        return cleaned[start : end + 1]
    return cleaned


def _shorten(text: str, max_length: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 3].rstrip()}..."


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
