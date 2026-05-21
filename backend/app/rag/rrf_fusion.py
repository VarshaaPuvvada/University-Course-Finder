from app.rag.course_repository import Course


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Course, float]]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[Course, float]]:
    scores: dict[str, float] = {}
    courses: dict[str, Course] = {}

    for ranked_list in ranked_lists:
        for rank, (course, _) in enumerate(ranked_list, start=1):
            scores[course.id] = scores.get(course.id, 0.0) + 1.0 / (k + rank)
            courses[course.id] = course

    fused = [(courses[course_id], score) for course_id, score in scores.items()]
    return sorted(fused, key=lambda item: item[1], reverse=True)[:top_k]

