from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.models.schemas import RecommendRequest
from app.multimodal.image_processor import ImageProcessor
from app.multimodal.pdf_parser import PDFParser
from app.routes.recommend import generate_recommendation


router = APIRouter(tags=["multimodal"])


@router.post("/upload-pdf")
async def upload_pdf(
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
        extracted_text = PDFParser().extract_text(await file.read())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

    return _recommend_from_extracted_text(
        extracted_text=extracted_text,
        current_skills=current_skills,
        student_level=student_level,
        career_goal=career_goal,
        top_k=_parse_top_k(top_k, topK),
        organizations=organizations,
        difficulties=difficulties,
        skill_categories=skill_categories,
        min_rating=min_rating,
        strict_difficulty=strict_difficulty,
        use_llm_judge=use_llm_judge,
        preferred_skills=preferred_skills,
        completed_courses=completed_courses,
        liked_courses=liked_courses,
        disliked_courses=disliked_courses,
        learner_progress=learner_progress,
        peer_group=peer_group,
    )


@router.post("/upload-image")
async def upload_image(
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
        extracted_text = ImageProcessor().extract_query_text(await file.read())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    return _recommend_from_extracted_text(
        extracted_text=extracted_text,
        current_skills=current_skills,
        student_level=student_level,
        career_goal=career_goal,
        top_k=_parse_top_k(top_k, topK),
        organizations=organizations,
        difficulties=difficulties,
        skill_categories=skill_categories,
        min_rating=min_rating,
        strict_difficulty=strict_difficulty,
        use_llm_judge=use_llm_judge,
        preferred_skills=preferred_skills,
        completed_courses=completed_courses,
        liked_courses=liked_courses,
        disliked_courses=disliked_courses,
        learner_progress=learner_progress,
        peer_group=peer_group,
    )


def _recommend_from_extracted_text(
    extracted_text: str,
    current_skills: str,
    student_level: str,
    career_goal: str,
    top_k: int,
    organizations: str,
    difficulties: str,
    skill_categories: str,
    min_rating: str,
    strict_difficulty: str,
    use_llm_judge: str,
    preferred_skills: str,
    completed_courses: str,
    liked_courses: str,
    disliked_courses: str,
    learner_progress: str,
    peer_group: str,
) -> dict:
    query = " ".join(extracted_text.split())
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="Upload did not contain enough readable text.")

    recommendation = generate_recommendation(
        RecommendRequest(
            query=query,
            current_skills=_split_skills(current_skills),
            student_level=student_level,
            career_goal=career_goal or None,
            top_k=top_k,
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
        "extracted_text": query,
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
