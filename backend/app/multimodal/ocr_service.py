import io


class OCRService:
    def extract_text(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow and pytesseract are required for image OCR.") from exc

        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image).strip()
