import os

from app.utils.env import load_backend_env


class GroqClient:
    def __init__(self, model: str | None = None) -> None:
        load_backend_env()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_DEFAULT_MODEL", "qwen-3-32b")

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
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

