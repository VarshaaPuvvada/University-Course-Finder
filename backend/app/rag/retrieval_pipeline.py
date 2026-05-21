from functools import lru_cache

from app.rag.bm25 import BM25Index
from app.rag.course_repository import Course, load_courses
from app.rag.embedding_service import OpenRouterEmbeddingService
from app.rag.pinecone_service import PineconeService
from app.rag.rrf_fusion import reciprocal_rank_fusion
from app.rag.semantic_search import LocalSemanticIndex
from app.reranker.bge_reranker import BGEReranker
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
) -> list[tuple[Course, float]]:
    normalized_query = normalize_query(query)
    course_by_id = {course.id: course for course in load_courses()}

    bm25_results = _bm25_index().search(normalized_query, top_k=30)
    semantic_results: list[tuple[Course, float]] = []

    embedding_service = OpenRouterEmbeddingService()
    pinecone_service = PineconeService()
    try:
        embedding = embedding_service.embed_text(normalized_query) if embedding_service.enabled else None
        if embedding and pinecone_service.enabled:
            pinecone_matches = pinecone_service.search(embedding, top_k=30)
            semantic_results = [
                (course_by_id[course_id], score)
                for course_id, score in pinecone_matches
                if course_id in course_by_id
            ]
    except Exception:
        semantic_results = []

    if not semantic_results:
        semantic_results = _semantic_index().search(normalized_query, top_k=30)

    fused = reciprocal_rank_fusion([semantic_results, bm25_results], top_k=30)
    reranked = BGEReranker().rerank(normalized_query, fused, top_k=30)

    max_reranked = max((score for _, score in reranked), default=1.0)

    personalized: list[tuple[Course, float]] = []
    for course, rerank_score in reranked:
        semantic_similarity = rerank_score / max_reranked
        final_score = (
            0.5 * semantic_similarity
            + 0.2 * _rating_score(course)
            + 0.15 * _difficulty_match(student_level, course)
            + 0.15 * _skill_overlap_score(normalized_query, current_skills, course)
        )
        personalized.append((course, final_score))

    return sorted(personalized, key=lambda item: item[1], reverse=True)[:top_k]
