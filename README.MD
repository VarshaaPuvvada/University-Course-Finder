# Intelligent University Course Finder

An AI-powered course recommendation system that accepts text, audio, PDF, and image inputs, extracts a learner goal, and recommends relevant Coursera-style university courses with explanations, prerequisite gaps, learning paths, and analytics.

The project is split into a FastAPI backend and a React/Vite frontend.

## Features

- Text-based course recommendations
- Audio input through Groq Whisper transcription
- PDF text extraction through `pypdf`
- Image OCR through Tesseract, with OpenRouter vision as the implemented fallback
- Hybrid retrieval using BM25 keyword search, Pinecone vector search, and TF-IDF semantic search as the implemented fallback
- Reciprocal Rank Fusion and reranking
- LangGraph multi-agent recommendation workflow
- Prerequisite gap detection with implemented prerequisite rules
- Learning path generation
- Career alignment summaries
- Analytics dashboard data
- RAGAS, DeepEval, and structural evaluation metrics
- LangSmith tracing

## Project Structure

```text
university-course-finder/
+-- README.MD
+-- audio_file_generate.py
+-- input_audio.mp3
+-- input_image.png
+-- input_text.pdf
+-- backend/
|   +-- requirements.txt
|   +-- evaluation_metrics.txt
|   +-- app/
|   |   +-- main.py
|   |   +-- agents/
|   |   |   +-- advisor_agent.py
|   |   |   +-- career_agent.py
|   |   |   +-- planner_agent.py
|   |   |   +-- retrieval_agent.py
|   |   |   +-- skill_gap_agent.py
|   |   |   +-- workflow.py
|   |   +-- evaluation/
|   |   |   +-- deepeval_eval.py
|   |   |   +-- groq_deepeval_model.py
|   |   |   +-- metrics_writer.py
|   |   |   +-- ragas_eval.py
|   |   |   +-- runner.py
|   |   +-- graph/
|   |   |   +-- prerequisite_validator.py
|   |   +-- guardrails/
|   |   |   +-- learning_guardrails.py
|   |   +-- learning/
|   |   |   +-- intelligence.py
|   |   +-- llm/
|   |   |   +-- groq_client.py
|   |   |   +-- model_router.py
|   |   +-- models/
|   |   |   +-- schemas.py
|   |   +-- multimodal/
|   |   |   +-- image_processor.py
|   |   |   +-- ocr_service.py
|   |   |   +-- pdf_parser.py
|   |   |   +-- whisper_service.py
|   |   +-- rag/
|   |   |   +-- bm25.py
|   |   |   +-- course_repository.py
|   |   |   +-- embedding_service.py
|   |   |   +-- index_courses.py
|   |   |   +-- pinecone_service.py
|   |   |   +-- retrieval_pipeline.py
|   |   |   +-- rrf_fusion.py
|   |   |   +-- semantic_search.py
|   |   +-- reranker/
|   |   |   +-- bge_reranker.py
|   |   +-- routes/
|   |   |   +-- analytics.py
|   |   |   +-- evaluation.py
|   |   |   +-- recommend.py
|   |   |   +-- speech.py
|   |   |   +-- upload.py
|   |   +-- utils/
|   |       +-- env.py
|   |       +-- text.py
|   |       +-- tracing.py
|   +-- data/
|   |   +-- raw/
|   |   |   +-- coursera_course_dataset_v2_no_null.csv
|   |   |   +-- coursera_course_dataset_v3.csv
|   |   +-- processed/
|   |       +-- cleaned_courses.csv
|   +-- notebooks/
|   |   +-- preprocessing.ipynb
|   +-- scripts/
|       +-- performance_test.py
+-- frontend/
    +-- index.html
    +-- package.json
    +-- package-lock.json
    +-- src/
        +-- App.jsx
        +-- main.jsx
        +-- api/
        |   +-- client.js
        +-- styles/
            +-- app.css
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React 18, Vite, lucide-react |
| Backend | FastAPI, Uvicorn |
| Data | Coursera course CSV files |
| PDF parsing | pypdf |
| Image OCR | Pillow, pytesseract, Tesseract executable, OpenRouter vision |
| Speech transcription | Groq Whisper API |
| Retrieval | BM25 keyword search, OpenRouter embeddings, Pinecone vector search, TF-IDF semantic fallback |
| Fusion/ranking | Reciprocal Rank Fusion, BGE reranker wrapper, personalized weighted scoring |
| Agents | LangGraph |
| Prerequisites | Rule-based prerequisite gap detection |
| Evaluation | RAGAS, DeepEval, structural metrics |
| Tracing | LangSmith |

## Data

The backend reads the processed course catalog from:

```text
backend/data/processed/cleaned_courses.csv
```

The processed file contains 3,612 course rows plus the header. Important columns include:

- `Title`
- `Organization`
- `Skills`
- `Ratings`
- `course_url`
- `course_description`
- `Review Count`
- `Difficulty`
- `Type`
- `Duration`

Raw source files are stored in:

```text
backend/data/raw/
```

## Backend Setup

From the repository root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `backend/.env` for API keys and service integrations. The file is intentionally ignored by git.

```env
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_VISION_MODEL=google/gemini-2.5-flash
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2

GROQ_API_KEY=
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3-turbo

PINECONE_API_KEY=
PINECONE_INDEX_NAME=university-courses
PINECONE_INDEX_DIMENSION=2048
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=courses

LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

```

### Environment Variables

- Text recommendations run with BM25 keyword search and TF-IDF semantic fallback.
- `GROQ_API_KEY` enables audio transcription and LLM-generated summaries.
- `OPENROUTER_API_KEY` enables OpenRouter embeddings and OpenRouter vision OCR.
- `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` enable Pinecone vector search over stored embeddings.
- `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` enable tracing.
- `SKIP_AUTO_EVALUATION=true` is recommended during development because automatic evaluation can add latency and may call external services.

## Run the Backend

From `backend/` with the virtual environment activated:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

From the repository root:

```powershell
cd frontend
npm install
```

Run the frontend:

```powershell
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend calls the backend at `http://127.0.0.1:8000` by default. To override it, set:

```env
VITE_API_BASE=http://127.0.0.1:8000
```

Build the frontend:

```powershell
npm run build
```

## User Inputs

The frontend keeps the visible form simple:

- Learning goal
- Current skills
- Student level
- Career goal
- Results count
- Audio upload
- PDF upload
- Image upload


Example learning goal:

```text
I want to learn machine learning for healthcare.
```

Example current skills:

```text
Python, statistics, SQL
```

## API Endpoints

### GET `/health`

Returns backend health status.

### POST `/recommend`

Main text recommendation endpoint.

Minimal request:

```json
{
  "query": "I want to learn AI for climate forecasting",
  "current_skills": ["Python", "Statistics"],
  "student_level": "beginner",
  "career_goal": "Climate AI Engineer",
  "top_k": 5
}
```

Response includes:

- normalized query
- recommendation list
- explanations
- prerequisite gaps
- learning path
- career alignment
- validation warnings
- agent handoffs
- skill graph summary

### POST `/speech-query`

Multipart audio upload endpoint.

Form fields:

- `file`
- `current_skills`
- `student_level`
- `career_goal`
- `top_k` (`topK` is also accepted for compatibility)

The backend transcribes audio with Groq Whisper, then runs the same recommendation pipeline.

### POST `/upload-pdf`

Multipart PDF upload endpoint.

Form fields:

- `file`
- `current_skills`
- `student_level`
- `career_goal`
- `top_k` (`topK` is also accepted for compatibility)

The backend extracts PDF text with `pypdf`, then runs the recommendation pipeline.

### POST `/upload-image`

Multipart image upload endpoint.

Form fields:

- `file`
- `current_skills`
- `student_level`
- `career_goal`
- `top_k` (`topK` is also accepted for compatibility)

The backend extracts image text with Tesseract when `tesseract.exe` is available. Missing Tesseract activates the implemented OpenRouter vision fallback when `OPENROUTER_API_KEY` is configured.

### GET `/analytics`

Returns catalog-level analytics:

- total indexed courses
- popular skills
- difficulty distribution
- top organizations
- success and completion estimates
- high-success courses
- skill graph overview

### POST `/evaluation`

Runs evaluation for a recommendation response and writes metrics to:

```text
backend/evaluation_metrics.txt
```

## Recommendation Pipeline

```text
Text / Audio / PDF / Image input
        |
        v
Text extraction and transcription layer
        |
        v
Query normalization
        |
        v
Candidate retrieval
  - BM25 keyword search
  - Pinecone vector search
  - TF-IDF semantic fallback
        |
        v
Reciprocal Rank Fusion
        |
        v
Reranking and personalized scoring
        |
        v
LangGraph agent workflow
  - Course retrieval agent
  - Skill gap agent
  - Learning path planner
  - Career alignment agent
  - Advisor agent
        |
        v
Prerequisite validation
        |
        v
Final recommendation response
        |
        v
Evaluation and tracing
```

## Retrieval Behavior

The retrieval layer includes a fallback path for environments that do not have external vector services configured:

1. Load courses from `cleaned_courses.csv`.
2. Filter candidates by metadata when filters are provided.
3. Run BM25 keyword search.
4. Run OpenRouter embeddings plus Pinecone vector search when service keys are configured.
5. Run TF-IDF semantic fallback when Pinecone results are unavailable.
6. Fuse results with RRF.
7. Rerank and score using:
   - semantic similarity
   - rating
   - difficulty fit
   - skill overlap
   - adaptive difficulty
   - estimated success rate
   - estimated completion rate
   - popularity
   - peer/career fit
   - preference signals

## Multimodal Notes

### PDF

PDF parsing uses `pypdf`.

### Image

Image OCR uses `pytesseract`, which requires the separate Tesseract executable for machine-level OCR.

Missing Tesseract activates the OpenRouter vision fallback when `OPENROUTER_API_KEY` is configured.

To point to a Windows Tesseract install manually:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Audio

Audio transcription uses the Groq API. Set `GROQ_API_KEY` before using `/speech-query`.

## Pinecone Indexing

To create embeddings and upsert the processed course catalog into Pinecone:

```powershell
cd backend
.\venv\Scripts\activate
python -m app.rag.index_courses
```

Required:

- `OPENROUTER_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

Missing Pinecone configuration activates BM25 keyword search plus TF-IDF semantic fallback.

## Evaluation

The system can write evaluation metrics to `backend/evaluation_metrics.txt`.

Evaluation includes:

- RAGAS metrics when dependencies and keys are available
- DeepEval metrics when dependencies and keys are available
- Structural metrics when external evaluators are unavailable

For faster development, set:

```env
SKIP_AUTO_EVALUATION=true
```

You can still call `/evaluation` manually.

## Performance Test

Run the load test script while the backend is running:

```powershell
cd backend
.\venv\Scripts\activate
python scripts/performance_test.py --base-url http://127.0.0.1:8000 --requests 30 --concurrency 5
```

For retrieval-only latency tests, use:

```env
SKIP_AUTO_EVALUATION=true
```

## Sample Assets

The repository includes sample input files for demos and manual testing:

- `input_text.pdf`
- `input_image.png`
- `input_audio.mp3`

`audio_file_generate.py` uses `gTTS` to generate a sample audio prompt. Install `gTTS` separately to regenerate the audio file:

```powershell
python -m pip install gTTS
python audio_file_generate.py
```

## Common Troubleshooting

### PDF upload returns a parsing dependency error

Install backend dependencies inside the backend virtual environment:

```powershell
cd backend
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Image upload says Tesseract is missing

Install Tesseract OCR on the machine. Missing Tesseract can be handled by configuring `OPENROUTER_API_KEY` for OpenRouter vision OCR.

### Audio upload fails

Set `GROQ_API_KEY` in `backend/.env`.

### Uploads always return 5 courses

Restart the backend server after code changes. The frontend sends both `top_k` and `topK`; the backend accepts either field and clamps the value between 1 and 20.

### Recommendations are slow

Set:

```env
SKIP_AUTO_EVALUATION=true
```

Evaluation and LLM summaries can add latency.

## Development Commands

Backend:

```powershell
cd backend
.\venv\Scripts\activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm run dev
```

Frontend build:

```powershell
cd frontend
npm run build
```

Python syntax check:

```powershell
cd backend
.\venv\Scripts\activate
python -m compileall app
```

## Notes for Reviewers

- The app remains usable when Pinecone is not configured because BM25 retrieval and TF-IDF semantic fallback are implemented.
- External keys unlock higher-quality speech, vision, embeddings, evaluation, and tracing.
- The visible frontend form intentionally hides advanced backend fields and sends safe defaults.
- `backend/.env`, `frontend/node_modules/`, and `frontend/dist/` are ignored by git.
