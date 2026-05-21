import math
from collections import Counter

from app.rag.course_repository import Course
from app.utils.text import tokenize


class BM25Index:
    def __init__(self, courses: list[Course], k1: float = 1.5, b: float = 0.75) -> None:
        self.courses = courses
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(course.combined_text) for course in courses]
        self.doc_lengths = [len(document) for document in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequencies = Counter(
            term for document in self.documents for term in set(document)
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[Course, float]]:
        query_terms = tokenize(query)
        scored: list[tuple[Course, float]] = []
        total_docs = len(self.documents)

        for index, course in enumerate(self.courses):
            score = 0.0
            doc_length = self.doc_lengths[index] or 1
            frequencies = self.term_frequencies[index]
            for term in query_terms:
                if term not in frequencies:
                    continue
                df = self.document_frequencies[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * (tf * (self.k1 + 1) / denominator)
            if score > 0:
                scored.append((course, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

