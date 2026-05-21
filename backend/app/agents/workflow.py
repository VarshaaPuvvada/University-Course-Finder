import warnings
from typing import TypedDict

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except ImportError:
    LangChainPendingDeprecationWarning = Warning

warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
    category=LangChainPendingDeprecationWarning,
)

from app.agents.advisor_agent import run_advisor_agent
from app.agents.career_agent import run_career_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.skill_gap_agent import run_skill_gap_agent
from app.rag.course_repository import Course


class RecommendationState(TypedDict):
    query: str
    current_skills: list[str]
    student_level: str
    career_goal: str | None
    top_k: int
    organizations: list[str]
    difficulties: list[str]
    skill_categories: list[str]
    min_rating: float | None
    strict_difficulty: bool
    ranked_courses: list[tuple[Course, float]]
    prerequisite_gaps: dict[str, list[str]]
    learning_path: list[str]
    career_alignment: str
    explanations: dict[str, str]


def run_recommendation_workflow(
    query: str,
    current_skills: list[str],
    student_level: str,
    career_goal: str | None,
    top_k: int,
    organizations: list[str] | None = None,
    difficulties: list[str] | None = None,
    skill_categories: list[str] | None = None,
    min_rating: float | None = None,
    strict_difficulty: bool = False,
) -> RecommendationState:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*allowed_objects.*",
                category=Warning,
            )
            from langgraph.graph import END, StateGraph
    except ImportError:
        return _run_sequential_workflow(
            query,
            current_skills,
            student_level,
            career_goal,
            top_k,
            organizations,
            difficulties,
            skill_categories,
            min_rating,
            strict_difficulty,
        )

    def retrieval_node(state: RecommendationState) -> RecommendationState:
        state["ranked_courses"] = run_retrieval_agent(
            state["query"],
            state["current_skills"],
            state["student_level"],
            state["top_k"],
            state["organizations"],
            state["difficulties"],
            state["skill_categories"],
            state["min_rating"],
            state["strict_difficulty"],
        )
        return state

    def skill_gap_node(state: RecommendationState) -> RecommendationState:
        state["prerequisite_gaps"] = run_skill_gap_agent(
            state["ranked_courses"], state["current_skills"]
        )
        return state

    def planner_node(state: RecommendationState) -> RecommendationState:
        state["learning_path"] = run_planner_agent(state["ranked_courses"])
        return state

    def career_node(state: RecommendationState) -> RecommendationState:
        state["career_alignment"] = run_career_agent(state["ranked_courses"], state["career_goal"])
        return state

    def advisor_node(state: RecommendationState) -> RecommendationState:
        state["explanations"] = run_advisor_agent(
            state["query"], state["ranked_courses"], state["prerequisite_gaps"]
        )
        return state

    graph = StateGraph(RecommendationState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("skill_gap", skill_gap_node)
    graph.add_node("planner", planner_node)
    graph.add_node("career", career_node)
    graph.add_node("advisor", advisor_node)
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "skill_gap")
    graph.add_edge("skill_gap", "planner")
    graph.add_edge("planner", "career")
    graph.add_edge("career", "advisor")
    graph.add_edge("advisor", END)
    return graph.compile().invoke(
        {
            "query": query,
            "current_skills": current_skills,
            "student_level": student_level,
            "career_goal": career_goal,
            "top_k": top_k,
            "organizations": organizations or [],
            "difficulties": difficulties or [],
            "skill_categories": skill_categories or [],
            "min_rating": min_rating,
            "strict_difficulty": strict_difficulty,
            "ranked_courses": [],
            "prerequisite_gaps": {},
            "learning_path": [],
            "career_alignment": "",
            "explanations": {},
        }
    )


def _run_sequential_workflow(
    query: str,
    current_skills: list[str],
    student_level: str,
    career_goal: str | None,
    top_k: int,
    organizations: list[str] | None = None,
    difficulties: list[str] | None = None,
    skill_categories: list[str] | None = None,
    min_rating: float | None = None,
    strict_difficulty: bool = False,
) -> RecommendationState:
    ranked_courses = run_retrieval_agent(
        query,
        current_skills,
        student_level,
        top_k,
        organizations,
        difficulties,
        skill_categories,
        min_rating,
        strict_difficulty,
    )
    prerequisite_gaps = run_skill_gap_agent(ranked_courses, current_skills)
    learning_path = run_planner_agent(ranked_courses)
    career_alignment = run_career_agent(ranked_courses, career_goal)
    explanations = run_advisor_agent(query, ranked_courses, prerequisite_gaps)
    return {
        "query": query,
        "current_skills": current_skills,
        "student_level": student_level,
        "career_goal": career_goal,
        "top_k": top_k,
        "organizations": organizations or [],
        "difficulties": difficulties or [],
        "skill_categories": skill_categories or [],
        "min_rating": min_rating,
        "strict_difficulty": strict_difficulty,
        "ranked_courses": ranked_courses,
        "prerequisite_gaps": prerequisite_gaps,
        "learning_path": learning_path,
        "career_alignment": career_alignment,
        "explanations": explanations,
    }
