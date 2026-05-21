from app.multimodal.ocr_service import OCRService


class ImageProcessor:
    def extract_query_text(self, image_bytes: bytes) -> str:
        return OCRService().extract_text(image_bytes)

