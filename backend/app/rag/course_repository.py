import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.utils.text import parse_skills


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "cleaned_courses.csv"


@dataclass(frozen=True)
class Course:
    id: str
    title: str
    organization: str
    skills: list[str]
    rating: float | None
    url: str | None
    description: str
    review_count: float
    difficulty: str
    course_type: str
    duration: str
    combined_text: str


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def load_courses() -> list[Course]:
    courses: list[Course] = []
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            skills = parse_skills(row.get("Skills"))
            title = row.get("Title", "").strip()
            organization = row.get("Organization", "").strip()
            difficulty = row.get("Difficulty", "").strip().lower() or "unknown"
            description = row.get("course_description", "").strip()
            combined_text = (
                f"Course: {title}. Organization: {organization}. "
                f"Skills: {', '.join(skills)}. Difficulty: {difficulty}. "
                f"Type: {row.get('Type', '').strip()}. Description: {description}"
            )
            rating_value = row.get("Ratings")
            courses.append(
                Course(
                    id=str(index),
                    title=title,
                    organization=organization,
                    skills=skills,
                    rating=_to_float(rating_value) if rating_value else None,
                    url=row.get("course_url"),
                    description=description,
                    review_count=_to_float(row.get("Review Count")),
                    difficulty=difficulty,
                    course_type=row.get("Type", "").strip(),
                    duration=row.get("Duration", "").strip(),
                    combined_text=combined_text,
                )
            )
    return courses

