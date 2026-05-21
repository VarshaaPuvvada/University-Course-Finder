from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2)
    current_skills: list[str] = Field(default_factory=list)
    student_level: str = "beginner"
    career_goal: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class CourseRecommendation(BaseModel):
    title: str
    organization: str
    difficulty: str
    rating: float | None
    url: str | None
    skills: list[str]
    explanation: str
    prerequisite_gaps: list[str]
    final_score: float


class RecommendResponse(BaseModel):
    normalized_query: str
    recommendations: list[CourseRecommendation]
    learning_path: list[str]
    career_alignment: str | None


class AnalyticsResponse(BaseModel):
    total_courses: int
    popular_skills: list[dict[str, str | int]]
    difficulty_distribution: dict[str, int]
    top_organizations: list[dict[str, str | int]]


class EvaluationRequest(BaseModel):
    recommendation: RecommendResponse


class EvaluationResponse(BaseModel):
    retrieval_metrics: dict[str, float]
    agent_metrics: dict[str, float]
    metrics_file: str | None = None
