import os

from app.utils.env import load_backend_env
from app.utils.tracing import trace_span


class GroqClient:
    def __init__(self, model: str | None = None) -> None:
        load_backend_env()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str | None:
        if not self.enabled:
            return None
        try:
            from groq import Groq
        except ImportError:
            return None

        client = Groq(api_key=self.api_key)
        try:
            with trace_span(
                "llm.groq.complete",
                run_type="llm",
                inputs={
                    "model": self.model,
                    "system_prompt_length": len(system_prompt),
                    "user_prompt_length": len(user_prompt),
                },
                metadata={"provider": "groq", "model": self.model},
            ):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
        except Exception:
            return None
        return response.choices[0].message.content
