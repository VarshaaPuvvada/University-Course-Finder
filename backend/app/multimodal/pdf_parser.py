class PDFParser:
    def extract_text(self, raw_pdf: bytes) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF parsing.") from exc

        text_parts: list[str] = []
        with fitz.open(stream=raw_pdf, filetype="pdf") as document:
            for page in document:
                text_parts.append(page.get_text("text"))
        return "\n".join(part.strip() for part in text_parts if part.strip())
