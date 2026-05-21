from app.evaluation.deepeval_eval import evaluate_agents
from app.evaluation.metrics_writer import write_evaluation_metrics
from app.evaluation.ragas_eval import evaluate_retrieval
from app.models.schemas import EvaluationResponse, RecommendResponse
from app.utils.tracing import trace_span


def evaluate_and_write(recommendation: RecommendResponse) -> EvaluationResponse:
    with trace_span(
        "evaluation.run_and_write",
        inputs={
            "normalized_query": recommendation.normalized_query,
            "recommendation_count": len(recommendation.recommendations),
        },
        metadata={"component": "evaluation"},
    ):
        retrieval_metrics = evaluate_retrieval(recommendation)
        agent_metrics = evaluate_agents(recommendation)
        metrics_path = write_evaluation_metrics(
            recommendation=recommendation,
            retrieval_metrics=retrieval_metrics,
            agent_metrics=agent_metrics,
            judge_metrics={},
        )
        return EvaluationResponse(
            retrieval_metrics=retrieval_metrics,
            agent_metrics=agent_metrics,
            metrics_file=str(metrics_path),
        )
