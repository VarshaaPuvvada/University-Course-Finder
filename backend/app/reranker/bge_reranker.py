from app.rag.course_repository import Course


class BGEReranker:
    """BGE cross-encoder hook for Phase 3 reranking."""

    def __init__(self) -> None:
        self.model_name = "BAAI/bge-reranker-base"

    def rerank(self, _query: str, candidates: list[tuple[Course, float]], top_k: int) -> list[tuple[Course, float]]:
        return candidates[:top_k]

