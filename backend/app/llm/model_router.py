from app.llm.groq_client import GroqClient


AGENT_MODELS = {
    "retrieval": "openai/gpt-oss-20b",
    "skill_gap": "openai/gpt-oss-120b",
    "planner": "openai/gpt-oss-120b",
    "career": "qwen/qwen3-32b",
    "advisor": "qwen/qwen3-32b",
}


def get_agent_llm(agent_name: str) -> GroqClient:
    return GroqClient(model=AGENT_MODELS.get(agent_name))

