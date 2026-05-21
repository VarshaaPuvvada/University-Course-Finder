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
from app.learning.intelligence import route_domain
from app.rag.course_repository import Course
from app.utils.tracing import trace_span


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
    preferred_skills: list[str]
    completed_courses: list[str]
    liked_courses: list[str]
    disliked_courses: list[str]
    learner_progress: float | None
    peer_group: str | None
    learner_domain: str
    agent_handoffs: list[str]
    agent_messages: list[dict[str, str]]
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
    preferred_skills: list[str] | None = None,
    completed_courses: list[str] | None = None,
    liked_courses: list[str] | None = None,
    disliked_courses: list[str] | None = None,
    learner_progress: float | None = None,
    peer_group: str | None = None,
) -> RecommendationState:
    learner_domain = route_domain(query, career_goal)
    agent_handoffs = [
        "course_retrieval_agent",
        "skill_gap_analysis_agent",
        f"{learner_domain}_domain_handoff",
        "learning_path_planning_agent",
        "career_alignment_agent",
        "learning_advisor_agent",
    ]
    with trace_span(
        "agents.recommendation_workflow",
        inputs={
            "query": query,
            "student_level": student_level,
            "career_goal": career_goal,
            "top_k": top_k,
        },
        metadata={"workflow": "langgraph"},
    ):
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
                preferred_skills,
                completed_courses,
                liked_courses,
                disliked_courses,
                learner_progress,
                peer_group,
            )

        def retrieval_node(state: RecommendationState) -> RecommendationState:
            with trace_span(
                "agent.retrieval",
                inputs={"query": state["query"], "top_k": state["top_k"]},
                metadata={"agent": "retrieval"},
            ):
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
                    state["preferred_skills"],
                    state["completed_courses"],
                    state["liked_courses"],
                    state["disliked_courses"],
                    state["learner_progress"],
                    state["career_goal"],
                )
                state["agent_messages"].append(
                    {
                        "from": "course_retrieval_agent",
                        "to": "skill_gap_analysis_agent",
                        "message": f"Retrieved {len(state['ranked_courses'])} personalized candidate courses.",
                    }
                )
            return state

        def skill_gap_node(state: RecommendationState) -> RecommendationState:
            with trace_span(
                "agent.skill_gap",
                inputs={"course_count": len(state["ranked_courses"])},
                metadata={"agent": "skill_gap"},
            ):
                state["prerequisite_gaps"] = run_skill_gap_agent(
                    state["ranked_courses"], state["current_skills"]
                )
                gap_count = sum(len(gaps) for gaps in state["prerequisite_gaps"].values())
                state["agent_messages"].append(
                    {
                        "from": "skill_gap_analysis_agent",
                        "to": f"{state['learner_domain']}_domain_handoff",
                        "message": f"Identified {gap_count} prerequisite gaps for domain-aware planning.",
                    }
                )
            return state

        def planner_node(state: RecommendationState) -> RecommendationState:
            with trace_span(
                "agent.planner",
                inputs={"course_count": len(state["ranked_courses"])},
                metadata={"agent": "planner"},
            ):
                state["learning_path"] = run_planner_agent(state["ranked_courses"])
                state["agent_messages"].append(
                    {
                        "from": "learning_path_planning_agent",
                        "to": "career_alignment_agent",
                        "message": f"Generated a {len(state['learning_path'])}-step learning path.",
                    }
                )
            return state

        def career_node(state: RecommendationState) -> RecommendationState:
            with trace_span(
                "agent.career",
                inputs={"career_goal": state["career_goal"]},
                metadata={"agent": "career"},
            ):
                state["career_alignment"] = run_career_agent(
                    state["ranked_courses"], state["career_goal"]
                )
                state["agent_messages"].append(
                    {
                        "from": "career_alignment_agent",
                        "to": "learning_advisor_agent",
                        "message": "Mapped recommendations to the learner career goal.",
                    }
                )
            return state

        def advisor_node(state: RecommendationState) -> RecommendationState:
            with trace_span(
                "agent.advisor",
                inputs={
                    "query": state["query"],
                    "course_count": len(state["ranked_courses"]),
                },
                metadata={"agent": "advisor"},
            ):
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
                "preferred_skills": preferred_skills or [],
                "completed_courses": completed_courses or [],
                "liked_courses": liked_courses or [],
                "disliked_courses": disliked_courses or [],
                "learner_progress": learner_progress,
                "peer_group": peer_group,
                "learner_domain": learner_domain,
                "agent_handoffs": agent_handoffs,
                "agent_messages": [],
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
    preferred_skills: list[str] | None = None,
    completed_courses: list[str] | None = None,
    liked_courses: list[str] | None = None,
    disliked_courses: list[str] | None = None,
    learner_progress: float | None = None,
    peer_group: str | None = None,
) -> RecommendationState:
    learner_domain = route_domain(query, career_goal)
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
        preferred_skills,
        completed_courses,
        liked_courses,
        disliked_courses,
        learner_progress,
        career_goal,
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
        "preferred_skills": preferred_skills or [],
        "completed_courses": completed_courses or [],
        "liked_courses": liked_courses or [],
        "disliked_courses": disliked_courses or [],
        "learner_progress": learner_progress,
        "peer_group": peer_group,
        "learner_domain": learner_domain,
        "agent_handoffs": [
            "course_retrieval_agent",
            "skill_gap_analysis_agent",
            f"{learner_domain}_domain_handoff",
            "learning_path_planning_agent",
            "career_alignment_agent",
            "learning_advisor_agent",
        ],
        "agent_messages": [
            {
                "from": "course_retrieval_agent",
                "to": "skill_gap_analysis_agent",
                "message": f"Retrieved {len(ranked_courses)} personalized candidate courses.",
            },
            {
                "from": "skill_gap_analysis_agent",
                "to": f"{learner_domain}_domain_handoff",
                "message": "Shared prerequisite gaps for domain-aware planning.",
            },
            {
                "from": "learning_path_planning_agent",
                "to": "career_alignment_agent",
                "message": f"Generated a {len(learning_path)}-step learning path.",
            },
        ],
        "ranked_courses": ranked_courses,
        "prerequisite_gaps": prerequisite_gaps,
        "learning_path": learning_path,
        "career_alignment": career_alignment,
        "explanations": explanations,
    }
