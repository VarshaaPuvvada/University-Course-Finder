from fastapi import APIRouter

from app.evaluation.deepeval_eval import evaluate_agents
from app.evaluation.metrics_writer import write_evaluation_metrics
from app.evaluation.ragas_eval import evaluate_retrieval
from app.models.schemas import EvaluationRequest, EvaluationResponse


router = APIRouter(tags=["evaluation"])


@router.post("/evaluation", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    retrieval_metrics = evaluate_retrieval(payload.recommendation)
    agent_metrics = evaluate_agents(payload.recommendation)
    metrics_path = write_evaluation_metrics(
        recommendation=payload.recommendation,
        retrieval_metrics=retrieval_metrics,
        agent_metrics=agent_metrics,
    )
    return EvaluationResponse(
        retrieval_metrics=retrieval_metrics,
        agent_metrics=agent_metrics,
        metrics_file=str(metrics_path),
    )
