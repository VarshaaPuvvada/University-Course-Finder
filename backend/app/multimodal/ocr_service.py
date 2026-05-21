import base64
import io
import json
import os
import urllib.error
import urllib.request

from app.utils.env import load_backend_env


class OCRService:
    def extract_text(self, image_bytes: bytes) -> str:
        load_backend_env()
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow and pytesseract are required for image OCR.") from exc

        image = Image.open(io.BytesIO(image_bytes))
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        try:
            return pytesseract.image_to_string(image).strip()
        except pytesseract.TesseractNotFoundError:
            return self._extract_with_openrouter(image_bytes, image.format)

    def _extract_with_openrouter(self, image_bytes: bytes, image_format: str | None) -> str:
        load_backend_env()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Tesseract is not installed and OPENROUTER_API_KEY is not configured for vision OCR."
            )

        model = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash")
        mime_type = _mime_type(image_format)
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all readable text from this image. "
                                "Return only the extracted text, with no commentary."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 1500,
        }

        request = urllib.request.Request(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "University Course Finder",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter vision OCR failed: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter vision OCR failed: {exc.reason}") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter vision OCR returned an unexpected response.") from exc


def _mime_type(image_format: str | None) -> str:
    format_to_mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
    }
    return format_to_mime.get((image_format or "").upper(), "image/png")
