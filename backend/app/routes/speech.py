from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.models.schemas import RecommendRequest
from app.multimodal.whisper_service import WhisperService
from app.routes.recommend import generate_recommendation


router = APIRouter(tags=["multimodal"])


@router.post("/speech-query")
async def speech_query(
    file: UploadFile,
    current_skills: str = Form(""),
    student_level: str = Form("beginner"),
    career_goal: str = Form(""),
    top_k: str = Form(""),
    topK: str = Form(""),
    organizations: str = Form(""),
    difficulties: str = Form(""),
    skill_categories: str = Form(""),
    min_rating: str = Form(""),
    strict_difficulty: str = Form("false"),
    use_llm_judge: str = Form("false"),
    preferred_skills: str = Form(""),
    completed_courses: str = Form(""),
    liked_courses: str = Form(""),
    disliked_courses: str = Form(""),
    learner_progress: str = Form(""),
    peer_group: str = Form(""),
) -> dict:
    try:
        transcript = await WhisperService().transcribe(file)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not transcribe audio: {exc}") from exc

    if len(transcript.strip()) < 2:
        raise HTTPException(status_code=400, detail="Audio transcription did not produce enough text.")

    recommendation = generate_recommendation(
        RecommendRequest(
            query=transcript,
            current_skills=_split_skills(current_skills),
            student_level=student_level,
            career_goal=career_goal or None,
            top_k=_parse_top_k(top_k, topK),
            organizations=_split_skills(organizations),
            difficulties=_split_skills(difficulties),
            skill_categories=_split_skills(skill_categories),
            min_rating=float(min_rating) if min_rating else None,
            strict_difficulty=_to_bool(strict_difficulty),
            use_llm_judge=_to_bool(use_llm_judge),
            preferred_skills=_split_skills(preferred_skills),
            completed_courses=_split_skills(completed_courses),
            liked_courses=_split_skills(liked_courses),
            disliked_courses=_split_skills(disliked_courses),
            learner_progress=float(learner_progress) if learner_progress else None,
            peer_group=peer_group or None,
        )
    )
    return {
        "status": "ok",
        "transcript": transcript,
        "recommendation": recommendation,
    }


def _split_skills(value: str) -> list[str]:
    return [skill.strip() for skill in value.split(",") if skill.strip()]


def _to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_top_k(*values: str) -> int:
    for value in values:
        if value:
            return min(max(int(value), 1), 20)
    return 5
