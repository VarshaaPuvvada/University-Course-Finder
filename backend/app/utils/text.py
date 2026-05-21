import ast
import re


_WHITESPACE_RE = re.compile(r"\s+")
_OCR_NOISE_RE = re.compile(r"[^\w\s.,:;!?+\-#/]")


def normalize_query(text: str) -> str:
    cleaned = _OCR_NOISE_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def parse_skills(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(skill).strip() for skill in parsed if str(skill).strip()]
    except (SyntaxError, ValueError):
        pass
    return [skill.strip() for skill in value.split(",") if skill.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#]+", normalize_query(text))

