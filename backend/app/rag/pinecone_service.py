import os
import time

from app.rag.course_repository import Course
from app.utils.env import load_backend_env


class PineconeService:
    def __init__(self) -> None:
        load_backend_env()
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "university-courses")
        self.dimension = int(os.getenv("PINECONE_INDEX_DIMENSION", "2048"))
        self.metric = os.getenv("PINECONE_METRIC", "cosine")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "courses")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.index_name)

    def _client(self):
        if not self.enabled:
            raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX_NAME are required.")
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RuntimeError("Install the Pinecone SDK with `pip install -r requirements.txt`.") from exc

        return Pinecone(api_key=self.api_key)

    def ensure_index(self) -> None:
        pc = self._client()
        if not pc.has_index(self.index_name):
            try:
                from pinecone import ServerlessSpec
            except ImportError as exc:
                raise RuntimeError("Install the Pinecone SDK with `pip install -r requirements.txt`.") from exc

            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                deletion_protection="disabled",
            )
        while not pc.describe_index(self.index_name).status["ready"]:
            time.sleep(2)

    def describe_stats(self) -> dict:
        if not self.enabled:
            return {}
        index = self._client().Index(self.index_name)
        try:
            stats = index.describe_index_stats()
        except Exception:
            return {}
        return dict(stats) if stats else {}

    def upsert_courses(
        self,
        courses: list[Course],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        if len(courses) != len(embeddings):
            raise ValueError("Course and embedding counts must match.")

        self.ensure_index()
        index = self._client().Index(self.index_name)
        total = 0
        for start in range(0, len(courses), batch_size):
            batch_courses = courses[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            vectors = [
                {
                    "id": course.id,
                    "values": [float(value) for value in embedding],
                    "metadata": {
                        "title": course.title,
                        "organization": course.organization,
                        "difficulty": course.difficulty,
                        "rating": course.rating or 0.0,
                        "url": course.url or "",
                    },
                }
                for course, embedding in zip(batch_courses, batch_embeddings)
            ]
            index.upsert(vectors=vectors, namespace=self.namespace)
            total += len(vectors)
        return total

    def search(self, embedding: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        if not self.enabled:
            return []
        index = self._client().Index(self.index_name)
        response = index.query(
            vector=[float(value) for value in embedding],
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=False,
        )
        return [(match["id"], float(match["score"])) for match in response.get("matches", [])]
