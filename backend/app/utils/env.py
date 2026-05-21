from pathlib import Path


def load_backend_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
