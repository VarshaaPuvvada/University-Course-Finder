from io import BytesIO


class PDFParser:
    def extract_text(self, raw_pdf: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            return self._extract_with_pymupdf(raw_pdf)

        reader = PdfReader(BytesIO(raw_pdf))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(part.strip() for part in text_parts if part.strip())

    def _extract_with_pymupdf(self, raw_pdf: bytes) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Install pypdf or PyMuPDF to enable PDF parsing.") from exc

        text_parts: list[str] = []
        with fitz.open(stream=raw_pdf, filetype="pdf") as document:
            for page in document:
                text_parts.append(page.get_text("text"))
        return "\n".join(part.strip() for part in text_parts if part.strip())
