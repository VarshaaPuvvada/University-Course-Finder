from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analytics import router as analytics_router
from app.routes.evaluation import router as evaluation_router
from app.routes.graph import router as graph_router
from app.routes.recommend import router as recommend_router
from app.routes.speech import router as speech_router
from app.routes.upload import router as upload_router


app = FastAPI(
    title="Intelligent University Course Finder",
    description="AI-powered multimodal course recommendation backend.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend_router)
app.include_router(speech_router)
app.include_router(upload_router)
app.include_router(analytics_router)
app.include_router(evaluation_router)
app.include_router(graph_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
