import math
from collections import Counter, defaultdict
from functools import lru_cache

from app.rag.course_repository import Course, load_courses
from app.utils.text import normalize_query, tokenize


DIFFICULTY_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "mixed": 2, "unknown": 2}


def course_success_rate(course: Course) -> float:
    rating = course.rating or 0.0
    rating_score = min(max(rating / 5.0, 0.0), 1.0)
    confidence = min(math.log1p(max(course.review_count, 0.0)) / math.log1p(1_000_000), 1.0)
    return round(0.68 * rating_score + 0.32 * confidence, 3)


def course_completion_rate(course: Course) -> float:
    success = course_success_rate(course)
    difficulty_penalty = {1: 0.05, 2: 0.12, 3: 0.2}.get(
        DIFFICULTY_RANK.get(course.difficulty, 2), 0.12
    )
    return round(min(max(success - difficulty_penalty + 0.12, 0.0), 1.0), 3)


@lru_cache(maxsize=1)
def popularity_percentiles() -> dict[str, float]:
    courses = load_courses()
    ordered = sorted(courses, key=lambda course: course.review_count)
    total = max(len(ordered) - 1, 1)
    return {course.id: round(index / total, 3) for index, course in enumerate(ordered)}


def preference_score(
    course: Course,
    *,
    preferred_skills: list[str],
    completed_courses: list[str],
    liked_courses: list[str],
    disliked_courses: list[str],
) -> float:
    title = course.title.lower()
    completed = {item.lower() for item in completed_courses}
    liked = {item.lower() for item in liked_courses}
    disliked = {item.lower() for item in disliked_courses}

    if any(item and item in title for item in completed):
        return -0.35
    if any(item and item in title for item in disliked):
        return -0.25

    preferred = {skill.lower() for skill in preferred_skills}
    course_skills = {skill.lower() for skill in course.skills}
    overlap = len(preferred & course_skills) / max(len(preferred), 1)
    liked_bonus = 0.18 if any(item and item in title for item in liked) else 0.0
    return min(0.32 * overlap + liked_bonus, 0.4)


def adaptive_difficulty_score(
    course: Course,
    *,
    student_level: str,
    current_skills: list[str],
    completed_courses: list[str],
    learner_progress: float | None,
) -> float:
    base_rank = DIFFICULTY_RANK.get(student_level.lower(), 1)
    progress = learner_progress
    if progress is None:
        progress = min((len(current_skills) * 0.08) + (len(completed_courses) * 0.12), 1.0)

    target_rank = base_rank
    if progress >= 0.72:
        target_rank = min(base_rank + 1, 3)
    elif progress <= 0.25:
        target_rank = max(base_rank, 1)

    course_rank = DIFFICULTY_RANK.get(course.difficulty, 2)
    distance = abs(course_rank - target_rank)
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.68
    return 0.25


def collaborative_peer_score(
    course: Course,
    *,
    current_skills: list[str],
    preferred_skills: list[str],
    career_goal: str | None,
) -> float:
    interest_terms = set()
    for value in [*current_skills, *preferred_skills, career_goal or ""]:
        interest_terms.update(tokenize(value))

    if not interest_terms:
        return 0.0

    related = related_skills_for_terms(tuple(sorted(interest_terms)))
    course_skills = {skill.lower() for skill in course.skills}
    if not course_skills:
        return 0.0

    overlap = len(course_skills & related) / len(course_skills)
    return min(overlap, 1.0)


@lru_cache(maxsize=256)
def related_skills_for_terms(terms: tuple[str, ...]) -> set[str]:
    graph = skill_graph()
    related: set[str] = set()
    term_set = set(terms)
    for skill, neighbors in graph["edges"].items():
        skill_terms = set(tokenize(skill))
        if skill_terms & term_set or skill.lower() in term_set:
            related.add(skill.lower())
            related.update(neighbor.lower() for neighbor, _ in neighbors[:8])
    return related


@lru_cache(maxsize=1)
def skill_graph() -> dict:
    courses = load_courses()
    skill_counts: Counter[str] = Counter()
    edge_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skill_ratings: dict[str, list[float]] = defaultdict(list)

    for course in courses:
        skills = [skill.strip() for skill in course.skills if skill.strip()]
        unique_skills = sorted(set(skills), key=str.lower)
        for skill in unique_skills:
            skill_counts[skill] += 1
            if course.rating is not None:
                skill_ratings[skill].append(course.rating)
        for index, left in enumerate(unique_skills):
            for right in unique_skills[index + 1 :]:
                edge_counts[left][right] += 1
                edge_counts[right][left] += 1

    nodes = [
        {
            "skill": skill,
            "course_count": count,
            "avg_rating": round(sum(skill_ratings[skill]) / len(skill_ratings[skill]), 2)
            if skill_ratings[skill]
            else None,
        }
        for skill, count in skill_counts.most_common()
    ]
    edges = {
        skill: neighbors.most_common(12)
        for skill, neighbors in edge_counts.items()
    }
    return {"nodes": nodes, "edges": edges}


def skill_graph_summary(skills: list[str], limit: int = 8) -> list[dict[str, object]]:
    graph = skill_graph()
    wanted = {skill.lower() for skill in skills}
    summaries: list[dict[str, object]] = []
    for node in graph["nodes"]:
        skill = str(node["skill"])
        if wanted and skill.lower() not in wanted:
            continue
        summaries.append(
            {
                **node,
                "related_skills": [
                    {"skill": related, "strength": count}
                    for related, count in graph["edges"].get(skill, [])[:6]
                ],
            }
        )
        if len(summaries) >= limit:
            break
    if summaries:
        return summaries
    return [
        {
            **node,
            "related_skills": [
                {"skill": related, "strength": count}
                for related, count in graph["edges"].get(str(node["skill"]), [])[:6]
            ],
        }
        for node in graph["nodes"][:limit]
    ]


def route_domain(query: str, career_goal: str | None = None) -> str:
    text = normalize_query(" ".join(part for part in [query, career_goal or ""] if part))
    domains = {
        "ai_ml": {"ai", "artificial", "machine", "learning", "deep", "neural", "prediction"},
        "data": {"data", "analytics", "sql", "statistics", "visualization", "database"},
        "finance": {"finance", "financial", "investment", "markets", "economics", "fintech"},
        "software": {"software", "programming", "web", "app", "developer", "cloud"},
        "cybersecurity": {"cybersecurity", "security", "network", "linux", "hacking"},
        "business": {"business", "management", "marketing", "product", "leadership"},
    }
    tokens = set(text.split())
    scores = {domain: len(tokens & keywords) for domain, keywords in domains.items()}
    best_domain, score = max(scores.items(), key=lambda item: item[1])
    return best_domain if score else "general"


def token_budget_text(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words]).rstrip(" ,;:")
    if truncated and truncated[-1] not in ".!?":
        truncated = f"{truncated}..."
    return truncated
