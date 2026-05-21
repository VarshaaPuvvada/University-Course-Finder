import os
import tempfile

from fastapi import UploadFile

from app.utils.env import load_backend_env


class WhisperService:
    def __init__(self) -> None:
        load_backend_env()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

    async def transcribe(self, file: UploadFile) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for speech transcription.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install groq to use speech transcription.") from exc

        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        try:
            client = Groq(api_key=self.api_key)
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    response_format="text",
                )
            return str(transcription).strip()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

