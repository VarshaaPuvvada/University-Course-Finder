from app.rag.course_repository import load_courses
from app.rag.embedding_service import OpenRouterEmbeddingService
from app.rag.pinecone_service import PineconeService


def index_courses(batch_size: int = 16) -> int:
    courses = load_courses()
    embedding_service = OpenRouterEmbeddingService()
    if not embedding_service.enabled:
        raise RuntimeError("OPENROUTER_API_KEY is required to generate course embeddings.")

    pinecone_service = PineconeService()
    if not pinecone_service.enabled:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX_NAME are required.")

    all_embeddings: list[list[float]] = []
    for start in range(0, len(courses), batch_size):
        batch = courses[start : start + batch_size]
        texts = [course.combined_text for course in batch]
        print(f"Embedding courses {start + 1}-{start + len(batch)} of {len(courses)}...")
        all_embeddings.extend(embedding_service.embed_texts(texts))

    return pinecone_service.upsert_courses(courses, all_embeddings)


if __name__ == "__main__":
    count = index_courses()
    print(f"Indexed {count} courses in Pinecone.")
