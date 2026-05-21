from app.rag.course_repository import Course


CAREER_KEYWORDS = {
    "data analyst": ["SQL", "Data Analysis", "Data Visualization", "Statistics"],
    "machine learning engineer": ["Python Programming", "Machine Learning", "Deep Learning"],
    "cybersecurity analyst": ["Network Security", "Linux", "Computer Security"],
    "cloud engineer": ["Cloud Computing", "DevOps", "Security"],
    "product manager": ["Product Management", "Leadership and Management", "Communication"],
}


def run_career_agent(
    ranked_courses: list[tuple[Course, float]],
    career_goal: str | None,
) -> str:
    if career_goal:
        goal = career_goal.lower()
        matched_track = next((track for track in CAREER_KEYWORDS if track in goal), career_goal)
        return f"These courses support the {matched_track} path by building job-relevant skills."

    top_skills = []
    seen = set()
    for course, _ in ranked_courses:
        for skill in course.skills[:3]:
            key = skill.lower()
            if key not in seen:
                top_skills.append(skill)
                seen.add(key)
            if len(top_skills) >= 4:
                break
        if len(top_skills) >= 4:
            break
    return f"Recommended career direction: roles using {', '.join(top_skills)}."

