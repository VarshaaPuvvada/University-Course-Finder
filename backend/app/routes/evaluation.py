from fastapi import APIRouter

from app.evaluation.runner import evaluate_and_write
from app.models.schemas import EvaluationRequest, EvaluationResponse


router = APIRouter(tags=["evaluation"])


@router.post("/evaluation", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    return evaluate_and_write(payload.recommendation)
