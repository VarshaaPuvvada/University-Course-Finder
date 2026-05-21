from functools import lru_cache

from app.rag.bm25 import BM25Index
from app.rag.course_repository import Course, load_courses
from app.rag.embedding_service import OpenRouterEmbeddingService
from app.rag.pinecone_service import PineconeService
from app.rag.rrf_fusion import reciprocal_rank_fusion
from app.rag.semantic_search import LocalSemanticIndex
from app.reranker.bge_reranker import BGEReranker
from app.learning.intelligence import (
    adaptive_difficulty_score,
    collaborative_peer_score,
    course_completion_rate,
    course_success_rate,
    popularity_percentiles,
    preference_score,
)
from app.utils.tracing import trace_span
from app.utils.text import normalize_query


DIFFICULTY_ORDER = {"beginner": 1, "intermediate": 2, "advanced": 3, "mixed": 2, "unknown": 2}


@lru_cache(maxsize=1)
def _bm25_index() -> BM25Index:
    return BM25Index(load_courses())


@lru_cache(maxsize=1)
def _semantic_index() -> LocalSemanticIndex:
    return LocalSemanticIndex(load_courses())


def _skill_overlap_score(query: str, current_skills: list[str], course: Course) -> float:
    query_terms = set(query.split())
    known_terms = {skill.lower() for skill in current_skills}
    course_skills = {skill.lower() for skill in course.skills}
    if not course_skills:
        return 0.0
    matches = {skill for skill in course_skills if skill in known_terms or any(term in skill for term in query_terms)}
    return len(matches) / len(course_skills)


def _difficulty_match(student_level: str, course: Course) -> float:
    student_rank = DIFFICULTY_ORDER.get(student_level.lower(), 1)
    course_rank = DIFFICULTY_ORDER.get(course.difficulty, 2)
    if course_rank == student_rank:
        return 1.0
    if course_rank == student_rank + 1:
        return 0.65
    if course_rank < student_rank:
        return 0.8
    return 0.2


def _rating_score(course: Course) -> float:
    if course.rating is None:
        return 0.0
    return min(max(course.rating / 5.0, 0.0), 1.0)


def retrieve_courses(
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
    with trace_span(
        "rag.retrieve_courses",
        inputs={"query": query, "student_level": student_level, "top_k": top_k},
        metadata={"component": "retrieval_pipeline"},
    ):
        normalized_query = normalize_query(query)
        course_by_id = {course.id: course for course in load_courses()}

        candidate_pool = [
            course
            for course in load_courses()
            if _passes_filters(
                course=course,
                student_level=student_level,
                organizations=organizations or [],
                difficulties=difficulties or [],
                skill_categories=skill_categories or [],
                min_rating=min_rating,
                strict_difficulty=strict_difficulty,
            )
        ]

        with trace_span(
            "rag.bm25_search",
            inputs={"normalized_query": normalized_query, "candidate_count": len(candidate_pool)},
            metadata={"retriever": "bm25"},
        ):
            bm25_results = [
                (course, score)
                for course, score in _bm25_index().search(normalized_query, top_k=80)
                if course in candidate_pool
            ][:30]
        semantic_results: list[tuple[Course, float]] = []

        embedding_service = OpenRouterEmbeddingService()
        pinecone_service = PineconeService()
        try:
            with trace_span(
                "rag.openrouter_embed",
                run_type="embedding",
                inputs={"text": normalized_query},
                metadata={"provider": "openrouter", "model": embedding_service.model},
            ):
                embedding = embedding_service.embed_text(normalized_query) if embedding_service.enabled else None
            if embedding and pinecone_service.enabled:
                with trace_span(
                    "rag.pinecone_search",
                    inputs={"top_k": 30, "index": pinecone_service.index_name},
                    metadata={"provider": "pinecone", "namespace": pinecone_service.namespace},
                ):
                    pinecone_matches = pinecone_service.search(embedding, top_k=30)
                semantic_results = [
                    (course_by_id[course_id], score)
                    for course_id, score in pinecone_matches
                    if course_id in course_by_id and course_by_id[course_id] in candidate_pool
                ]
        except Exception:
            semantic_results = []

        if not semantic_results:
            with trace_span(
                "rag.local_semantic_search",
                inputs={"normalized_query": normalized_query},
                metadata={"retriever": "local_tfidf"},
            ):
                semantic_results = [
                    (course, score)
                    for course, score in _semantic_index().search(normalized_query, top_k=80)
                    if course in candidate_pool
                ][:30]

        with trace_span(
            "rag.fusion_and_rerank",
            inputs={
                "semantic_count": len(semantic_results),
                "bm25_count": len(bm25_results),
            },
        ):
            fused = reciprocal_rank_fusion([semantic_results, bm25_results], top_k=30)
            reranked = BGEReranker().rerank(normalized_query, fused, top_k=30)

        max_reranked = max((score for _, score in reranked), default=1.0) or 1.0
        popularity = popularity_percentiles()

        personalized: list[tuple[Course, float]] = []
        for course, rerank_score in reranked:
            semantic_similarity = rerank_score / max_reranked
            preference = preference_score(
                course,
                preferred_skills=preferred_skills or [],
                completed_courses=completed_courses or [],
                liked_courses=liked_courses or [],
                disliked_courses=disliked_courses or [],
            )
            adaptive_fit = adaptive_difficulty_score(
                course,
                student_level=student_level,
                current_skills=current_skills,
                completed_courses=completed_courses or [],
                learner_progress=learner_progress,
            )
            peer_score = collaborative_peer_score(
                course,
                current_skills=current_skills,
                preferred_skills=preferred_skills or [],
                career_goal=career_goal,
            )
            success = course_success_rate(course)
            completion = course_completion_rate(course)
            popularity_score = popularity.get(course.id, 0.0)
            final_score = (
                0.34 * semantic_similarity
                + 0.14 * _rating_score(course)
                + 0.12 * _difficulty_match(student_level, course)
                + 0.1 * _skill_overlap_score(normalized_query, current_skills, course)
                + 0.1 * adaptive_fit
                + 0.08 * success
                + 0.05 * completion
                + 0.04 * popularity_score
                + 0.03 * peer_score
                + preference
            )
            personalized.append((course, final_score))

        return sorted(personalized, key=lambda item: item[1], reverse=True)[:top_k]


def _passes_filters(
    course: Course,
    student_level: str,
    organizations: list[str],
    difficulties: list[str],
    skill_categories: list[str],
    min_rating: float | None,
    strict_difficulty: bool,
) -> bool:
    if organizations:
        allowed_orgs = {organization.lower() for organization in organizations}
        if course.organization.lower() not in allowed_orgs:
            return False

    if difficulties:
        allowed_difficulties = {difficulty.lower() for difficulty in difficulties}
        if course.difficulty.lower() not in allowed_difficulties:
            return False

    if strict_difficulty and not difficulties:
        if course.difficulty.lower() != student_level.lower():
            return False

    if min_rating is not None and (course.rating is None or course.rating < min_rating):
        return False

    if skill_categories:
        required_skills = {skill.lower() for skill in skill_categories}
        course_skills = {skill.lower() for skill in course.skills}
        if not required_skills & course_skills:
            return False

    return True
