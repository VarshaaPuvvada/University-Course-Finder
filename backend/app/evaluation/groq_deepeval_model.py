import asyncio

from deepeval.models.base_model import DeepEvalBaseLLM

from app.llm.groq_client import GroqClient


class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self) -> None:
        super().__init__(model_name="groq-llama-3.3-70b-versatile")

    def load_model(self) -> GroqClient:
        return GroqClient()

    def generate(self, prompt: str, *args, **kwargs) -> str:
        response = self.model.complete(
            system_prompt=(
                "You are an evaluator for an academic course recommendation system. "
                "Follow the requested output format exactly."
            ),
            user_prompt=str(prompt),
        )
        return response or "{}"

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        return "groq/llama-3.3-70b-versatile"
