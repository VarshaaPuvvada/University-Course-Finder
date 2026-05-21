from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2)
    current_skills: list[str] = Field(default_factory=list)
    student_level: str = "beginner"
    career_goal: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    organizations: list[str] = Field(default_factory=list)
    difficulties: list[str] = Field(default_factory=list)
    skill_categories: list[str] = Field(default_factory=list)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    strict_difficulty: bool = False
    use_llm_judge: bool = False
    preferred_skills: list[str] = Field(default_factory=list)
    completed_courses: list[str] = Field(default_factory=list)
    liked_courses: list[str] = Field(default_factory=list)
    disliked_courses: list[str] = Field(default_factory=list)
    learner_progress: float | None = Field(default=None, ge=0, le=1)
    peer_group: str | None = None


class CourseRecommendation(BaseModel):
    title: str
    organization: str
    difficulty: str
    rating: float | None
    url: str | None
    skills: list[str]
    description: str
    course_type: str
    duration: str
    review_count: float
    explanation: str
    prerequisite_gaps: list[str]
    final_score: float
    llm_enhanced: bool = False
    success_rate: float | None = None
    completion_rate: float | None = None
    popularity_percentile: float | None = None
    matched_skills: list[str] = Field(default_factory=list)
    peer_recommendation_score: float | None = None


class RecommendResponse(BaseModel):
    normalized_query: str
    recommendations: list[CourseRecommendation]
    learning_path: list[str]
    career_alignment: str | None
    advisor_summary: str | None = None
    validation_warnings: list[str] = Field(default_factory=list)
    learner_domain: str | None = None
    agent_handoffs: list[str] = Field(default_factory=list)
    agent_messages: list[dict[str, str]] = Field(default_factory=list)
    skill_graph: list[dict] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    total_courses: int
    popular_skills: list[dict[str, str | int]]
    difficulty_distribution: dict[str, int]
    top_organizations: list[dict[str, str | int]]
    success_rate_by_difficulty: dict[str, float] = Field(default_factory=dict)
    completion_rate_by_difficulty: dict[str, float] = Field(default_factory=dict)
    high_success_courses: list[dict[str, str | float]] = Field(default_factory=list)
    skill_graph_overview: list[dict] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    recommendation: RecommendResponse


class EvaluationResponse(BaseModel):
    retrieval_metrics: dict[str, float]
    agent_metrics: dict[str, float]
    judge_metrics: dict[str, float] = Field(default_factory=dict)
    metrics_file: str | None = None
