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
    top_k: int = Form(5),
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
        top_k=top_k,
    )


@router.post("/upload-image")
async def upload_image(
    file: UploadFile,
    current_skills: str = Form(""),
    student_level: str = Form("beginner"),
    career_goal: str = Form(""),
    top_k: int = Form(5),
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
        top_k=top_k,
    )


def _recommend_from_extracted_text(
    extracted_text: str,
    current_skills: str,
    student_level: str,
    career_goal: str,
    top_k: int,
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
        )
    )
    return {
        "status": "ok",
        "extracted_text": query,
        "recommendation": recommendation,
    }


def _split_skills(value: str) -> list[str]:
    return [skill.strip() for skill in value.split(",") if skill.strip()]
