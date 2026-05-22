from app.rag.course_repository import Course


BASE_PREREQUISITES = {
    "artificial intelligence": ["python programming", "machine learning"],
    "deep learning": ["python programming", "machine learning"],
    "machine learning": ["python programming", "statistics"],
    "data science": ["python programming", "statistics"],
    "cybersecurity": ["network security", "linux"],
}


def find_prerequisite_gaps(course: Course, current_skills: list[str]) -> list[str]:
    known = {skill.lower() for skill in current_skills}
    course_skills = {skill.lower() for skill in course.skills}
    required = expand_required_skills(course_skills)
    return sorted(skill for skill in required if skill not in known and skill not in course_skills)


def expand_required_skills(skills: set[str]) -> set[str]:
    required: set[str] = set()
    for skill in skills:
        required.update(BASE_PREREQUISITES.get(skill.lower(), []))
    return required


def build_learning_path(recommended_courses: list[Course]) -> list[str]:
    path: list[str] = []
    seen: set[str] = set()
    for course in recommended_courses:
        for skill in course.skills[:3]:
            key = skill.lower()
            if key not in seen:
                path.append(skill)
                seen.add(key)
    return path[:8]


def validate_sequence(learning_path: list[str]) -> list[str]:
    priority = {
        "python programming": 0,
        "statistics": 1,
        "data analysis": 2,
        "machine learning": 3,
        "deep learning": 4,
        "artificial intelligence": 5,
    }
    return sorted(learning_path, key=lambda skill: priority.get(skill.lower(), 10))
