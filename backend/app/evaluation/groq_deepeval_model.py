import asyncio
import json

from deepeval.models.base_model import DeepEvalBaseLLM

from app.llm.groq_client import GroqClient


class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self) -> None:
        super().__init__(model_name="groq-llama-3.3-70b-versatile")

    def load_model(self) -> GroqClient:
        return GroqClient()

    def generate(self, prompt: str, *args, **kwargs):
        response = self.model.complete(
            system_prompt=(
                "You are an evaluator for an academic course recommendation system. "
                "Follow the requested output format exactly."
            ),
            user_prompt=str(prompt),
        )
        response = response or "{}"
        schema = kwargs.get("schema")
        if schema is None:
            return response

        json_text = _extract_json(response)
        try:
            return schema.model_validate_json(json_text)
        except Exception:
            return schema.model_validate(json.loads(json_text))

    async def a_generate(self, prompt: str, *args, **kwargs):
        return await asyncio.to_thread(self.generate, prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        return "groq/llama-3.3-70b-versatile"


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        return cleaned[start : end + 1]
    return cleaned
