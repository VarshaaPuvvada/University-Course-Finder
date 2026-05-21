import math
from collections import Counter

from app.rag.course_repository import Course
from app.utils.text import tokenize


class LocalSemanticIndex:
    """Local vector-space semantic search for Phase 1 hybrid retrieval."""

    def __init__(self, courses: list[Course]) -> None:
        self.courses = courses
        self.documents = [tokenize(course.combined_text) for course in courses]
        self.document_frequencies = Counter(
            term for document in self.documents for term in set(document)
        )
        self.total_documents = len(self.documents)
        self.course_vectors = [self._vectorize_terms(document) for document in self.documents]
        self.course_norms = [self._norm(vector) for vector in self.course_vectors]

    def search(self, query: str, top_k: int = 20) -> list[tuple[Course, float]]:
        query_vector = self._vectorize_terms(tokenize(query))
        query_norm = self._norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[tuple[Course, float]] = []
        for course, course_vector, course_norm in zip(
            self.courses, self.course_vectors, self.course_norms
        ):
            if course_norm == 0:
                continue
            score = self._cosine(query_vector, course_vector, query_norm, course_norm)
            if score > 0:
                scored.append((course, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _vectorize_terms(self, terms: list[str]) -> dict[str, float]:
        counts = Counter(terms)
        vector: dict[str, float] = {}
        for term, count in counts.items():
            df = self.document_frequencies.get(term, 0)
            idf = math.log((1 + self.total_documents) / (1 + df)) + 1
            vector[term] = (1 + math.log(count)) * idf
        return vector

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine(
        left: dict[str, float],
        right: dict[str, float],
        left_norm: float,
        right_norm: float,
    ) -> float:
        shared_terms = set(left) & set(right)
        dot_product = sum(left[term] * right[term] for term in shared_terms)
        return dot_product / (left_norm * right_norm)
