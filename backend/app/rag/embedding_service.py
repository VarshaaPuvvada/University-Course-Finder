import json
import os
import urllib.request

from app.utils.env import load_backend_env


class OpenRouterEmbeddingService:
    """OpenAI-compatible OpenRouter client for NVIDIA Nemotron embeddings."""

    def __init__(self) -> None:
        load_backend_env()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def embed_text(self, text: str) -> list[float] | None:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return []

        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]
