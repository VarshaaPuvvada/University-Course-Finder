from app.rag.course_repository import Course
from app.utils.text import tokenize


class BGEReranker:
    """Cross-encoder reranker with a deterministic fallback.

    If sentence-transformers is installed in the runtime, this uses the real
    BAAI/bge-reranker-base cross encoder. In lighter environments, it falls
    back to a query-document interaction score so reranking remains functional.
    """

    def __init__(self) -> None:
        self.model_name = "BAAI/bge-reranker-base"
        self._model = None

    def rerank(self, query: str, candidates: list[tuple[Course, float]], top_k: int) -> list[tuple[Course, float]]:
        if not candidates:
            return []

        cross_scores = self._cross_encoder_scores(query, [course for course, _ in candidates])
        if cross_scores is None:
            cross_scores = [
                self._interaction_score(query, course)
                for course, _ in candidates
            ]

        max_base = max((score for _, score in candidates), default=1.0) or 1.0
        normalized_cross = _normalize(cross_scores)
        reranked = []
        for index, (course, base_score) in enumerate(candidates):
            blended = 0.58 * normalized_cross[index] + 0.42 * (base_score / max_base)
            reranked.append((course, blended))
        return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_k]

    def _cross_encoder_scores(self, query: str, courses: list[Course]) -> list[float] | None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            return None

        try:
            if self._model is None:
                self._model = CrossEncoder(self.model_name)
            pairs = [(query, course.combined_text[:1800]) for course in courses]
            return [float(score) for score in self._model.predict(pairs)]
        except Exception:
            return None

    @staticmethod
    def _interaction_score(query: str, course: Course) -> float:
        query_terms = set(tokenize(query))
        doc_terms = set(tokenize(course.combined_text))
        skill_terms = {term for skill in course.skills for term in tokenize(skill)}
        title_terms = set(tokenize(course.title))
        if not query_terms:
            return 0.0
        doc_overlap = len(query_terms & doc_terms) / len(query_terms)
        skill_overlap = len(query_terms & skill_terms) / len(query_terms)
        title_overlap = len(query_terms & title_terms) / len(query_terms)
        return 0.45 * doc_overlap + 0.35 * skill_overlap + 0.2 * title_overlap


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]
