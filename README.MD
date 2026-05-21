# Intelligent University Course Finder – Complete Architecture & Implementation Plan

# 1. Project Goal

Build an AI-powered multimodal course recommendation system that:

* understands user learning goals
* retrieves relevant university/coursera courses
* identifies prerequisite gaps
* generates learning paths
* aligns recommendations with careers
* supports multimodal inputs
* uses hybrid retrieval + multi-agent orchestration

The system should support:

* text queries
* speech input
* image/screenshot input
* PDF/document input

Example query:

> “I want to learn how AI can predict stock prices.”

The system should:

* retrieve relevant courses
* identify missing skills
* validate prerequisites
* generate structured learning paths
* explain recommendations

---

# 2. Final Recommended Technology Stack

| Component          | Recommended Tool                       |
| ------------------ | -------------------------------------- |
| Frontend           | React + Vite              |
| Backend            | FastAPI                                |
| Speech-to-text     | Whisper                                |
| PDF Parsing        | PyMuPDF                                |
| OCR                | Tesseract OCR                          |
| Embeddings         | OpenAI text-embedding-3-small or large |
| Vector DB          | Pinecone                               |
| Keyword Search     | Local BM25                             |
| Hybrid Retrieval   | BM25 + Semantic Search                 |
| Fusion             | Reciprocal Rank Fusion (RRF)           |
| Reranking          | BGE Reranker / Cohere Rerank           |
| Multi-Agent System | LangGraph                              |
| Knowledge Graph    | Neo4j                                  |
| RAG Evaluation     | RAGAS                                  |
| Agent Evaluation   | DeepEval                               |
| Tracing            | LangSmith                              |

---

# 3. Why These Tools Were Chosen

## Whisper

Chosen because:

* open source
* multilingual
* works offline
* high accuracy
* free

Alternatives considered:

* Google Speech-to-Text
* AssemblyAI

Whisper was preferred due to cost and flexibility.

---

## PyMuPDF

Chosen because:

* fast
* layout aware
* better for RAG ingestion
* handles complex PDFs better than PyPDF

PyPDF is mainly useful for simple PDF manipulation.

---

## Pinecone

Chosen because:

* easy vector database setup
* managed infrastructure
* excellent semantic retrieval support

---

## LangGraph

Chosen over CrewAI and AutoGen because:

* stateful workflows
* graph-based orchestration
* proper agent handoffs
* better multi-agent control
* suitable for structured pipelines

CrewAI is easier but weaker for complex orchestration.
AutoGen is conversation-oriented and less deterministic.

---

## Neo4j

Chosen because:

* prerequisites are naturally graph relationships
* enables skill dependency traversal
* helps learning path generation
* enables prerequisite validation guardrails

---

## RAGAS + DeepEval

RAGAS:

* evaluates retrieval quality
* measures faithfulness/context relevance

DeepEval:

* evaluates agent reasoning
* hallucination detection
* LLM-as-judge support

---

# 4. Final High-Level Architecture

```text
Multimodal Input
(text/speech/pdf/image)
        ↓
Text Extraction Layer
(Whisper / OCR / PyMuPDF)
        ↓
Query Normalization
        ↓
Embedding Generation
        ↓
Hybrid Retrieval
(semantic + BM25)
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Cross Encoder Reranking
        ↓
LangGraph Multi-Agent Pipeline
        ↓
Neo4j Prerequisite Validation
        ↓
Learning Path Generation
        ↓
Final Recommendation
        ↓
Evaluation + Tracing
(RAGAS + DeepEval + LangSmith)
```

---

# 5. Understanding the Two Inputs in the System

The system has TWO main data streams.

---

# A. User Input Stream

Dynamic user query.

Input can be:

* text
* speech
* image
* screenshot
* PDF

Example:

> “I want to learn AI for healthcare.”

Pipeline:

```text
Speech → Whisper → text
Image/PDF → OCR → text
Text → normalization
↓
embedding generation
↓
hybrid retrieval query
```

---

# B. Course Dataset Stream

Static knowledge base.

Dataset fields:

* course_name
* description
* skills
* difficulty_level
* organization
* rating

This data becomes:

* vector embeddings
* BM25 searchable index
* prerequisite graph source

---

# 6. Dataset Choice

Preferred dataset:

Kaggle Coursera Dataset.

Reason:

* easier CSV handling
* simpler preprocessing
* better pandas integration
* cleaner ingestion pipeline

HuggingFace dataset version is more useful for training pipelines and dataset APIs.

---

# 7. Preprocessing Strategy

Modern embedding-based RAG systems should NOT use heavy traditional NLP preprocessing.

---

# User Query Preprocessing

Recommended:

* lowercase
* whitespace normalization
* OCR cleanup
* noise removal

Avoid:

* aggressive stopword removal
* stemming
* heavy lemmatization

Reason:
Embedding models understand natural language better than heavily cleaned tokens.

Example:

GOOD:

```text
I want to learn machine learning
```

BAD:

```text
want learn machin learn
```

---

# Course Data Preprocessing

Recommended:

* remove duplicates
* handle missing values
* normalize metadata
* combine fields into rich text

Example:

```text
Course: Machine Learning.
Skills: Python, regression, classification.
Difficulty: beginner.
```

---

# 8. Hybrid Retrieval System

The system uses BOTH:

1. Semantic Search
2. Keyword Search

---

# Semantic Search

Uses:

* embeddings
* Pinecone

Purpose:

* understand meaning
* retrieve conceptually related courses

---

# Keyword Search

Uses:

* BM25 

Purpose:

* exact term matching
* skill-specific retrieval

---

# Why Hybrid Retrieval Is Important

Semantic search alone may miss:

* exact technologies
* specific skill names

BM25 alone may miss:

* semantic meaning
* conceptual similarity

Hybrid retrieval combines both strengths.

---

# 9. Reciprocal Rank Fusion (RRF)

Hybrid retrieval results are merged using:

Reciprocal Rank Fusion.

Reason:

* more stable than raw score merging
* uses ranking positions instead of incompatible scores
* commonly used in production retrieval systems

Pipeline:

```text
Semantic Search Results
+
BM25 Results
↓
RRF Fusion
↓
Merged Candidates
```

---

# 10. Reranking Layer

After hybrid retrieval, reranking improves final recommendation quality.

Tools:

* BGE reranker
* Cohere rerank

Purpose:

* read query and course together
* improve final relevance ordering

---

# 11. Personalized Ranking

The ranking stage includes learning preference modeling.

This happens AFTER retrieval.

Pipeline:

```text
Hybrid Retrieval
↓
Candidate Courses
↓
Weighted Scoring + Reranking
↓
Final Courses
```

---

# Weighted Scoring Formula

Example:

```text
final_score =
0.5 * semantic_similarity +
0.2 * rating_score +
0.15 * difficulty_match +
0.15 * skill_overlap
```

---

# Features Used

## Semantic Similarity

Cosine similarity between:

* query embedding
* course embedding

## Rating Score

Higher-rated courses receive ranking boosts.

## Difficulty Match

Matches course difficulty to student level.

## Skill Overlap

Measures overlap between:

* user skills
* course skills

---

# 12. Multi-Agent System

Implemented using LangGraph.

---

# Agents Required

## 1. Course Retrieval Agent

Retrieves relevant courses.

---

## 2. Skill Gap Analysis Agent

Identifies missing prerequisite skills.

---

## 3. Learning Path Planning Agent

Creates structured learning sequences.

---

## 4. Career Alignment Agent

Maps courses to career tracks.

---

## 5. Learning Advisor Agent

Generates final recommendation summaries.

---

# Why LangGraph Was Preferred

Compared with CrewAI and AutoGen:

LangGraph was preferred because:

* graph-based orchestration
* persistent state
* deterministic execution
* proper agent handoffs
* suitable for structured AI pipelines

CrewAI:

* easier
* good for demos
* weaker orchestration control

AutoGen:

* conversation-focused
* less deterministic

---

# 13. Knowledge Graph and Neo4j

Neo4j is used to represent:

* skills
* prerequisites
* course relationships
* learning progression

---

# Example Graph

```text
Python → Machine Learning → Deep Learning
```

---

# Why Neo4j Helps

Enables:

* prerequisite validation
* learning path generation
* skill gap analysis
* dependency traversal

---

# 14. Learning Objective Validation Guardrails

Guardrails prevent:

* invalid recommendations
* unrealistic course paths
* prerequisite violations
* inappropriate difficulty jumps

---

# Important Guardrails

## 1. Prerequisite Validation Guardrail

Checks whether student has required foundational knowledge.

Implemented using Neo4j graph traversal.

---

## 2. Difficulty Validation Guardrail

Prevents beginner students from receiving highly advanced recommendations.

---

## 3. Learning Objective Alignment Guardrail

Ensures recommendations align with the student’s stated goal.

---

## 4. Sequence Validation Guardrail

Ensures learning paths follow valid order.

Example:

GOOD:

```text
Python → ML → Deep Learning
```

BAD:

```text
Deep Learning → Python
```

---

# 15. Evaluation Layer

---

# RAGAS

Used for:

* retrieval evaluation
* context precision
* faithfulness
* recall

---

# DeepEval

Used for:

* agent evaluation
* hallucination detection
* reasoning evaluation
* LLM-as-judge

---

# 16. LangSmith Tracing

Used for:

* observability
* debugging
* prompt tracing
* latency tracking
* token tracking

Very useful during development and demos.

---

# 17. FastAPI Backend Design

FastAPI acts as the AI orchestration backend.

---

# Suggested API Endpoints

## POST /recommend

Main recommendation endpoint.

---

## POST /speech-query

Audio input endpoint.

---

## POST /upload-pdf

PDF upload endpoint.

---

## POST /upload-image

Image upload endpoint.

---

## GET /analytics

Returns:

* popular courses
* trends
* statistics

---

# 18. React Frontend Design

Suggested pages:

## Home/Search Page

* text input
* upload support
* speech input

---

## Recommendation Page

Shows:

* recommended courses
* ratings
* prerequisite alerts
* explanations

---

## Learning Path Visualization

Can use:

* React Flow
* D3.js

---

## Analytics Dashboard

Shows:

* trending courses
* completion trends
* popular skills

---

# 19. Suggested Backend Folder Structure

```text
backend/
│
├── app/
│   ├── main.py
│   ├── routes/
│   ├── agents/
│   ├── rag/
│   ├── graph/
│   ├── reranker/
│   ├── evaluation/
│   ├── utils/
│   └── models/
│
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

---

# 20. Suggested Frontend Folder Structure

```text
frontend/
│
├── src/
│   ├── pages/
│   ├── components/
│   ├── api/
│   ├── hooks/
│   └── utils/
│
├── public/
└── package.json
```

---

# 21. Recommended Implementation Order

Do NOT start with agents.

Correct order:

## Phase 1

* dataset ingestion
* preprocessing
* embeddings
* Pinecone semantic setup + local BM25 setup

---

## Phase 2

* basic hybrid retrieval API

---

## Phase 3

* reranking
* weighted scoring

---

## Phase 4

* Neo4j prerequisite graph

---

## Phase 5

* LangGraph agents

---

## Phase 6

* React frontend

---

## Phase 7

* evaluation
* tracing
* optimization

---

# 22. Final Core Pipeline

```text
User Input
(text/speech/pdf/image)
        ↓
Text Extraction
        ↓
Normalization
        ↓
Embedding Generation
        ↓
Hybrid Retrieval
(Pinecone + BM25)
        ↓
RRF Fusion
        ↓
Cross Encoder Reranking
        ↓
LangGraph Agent Pipeline
        ↓
Neo4j Prerequisite Validation
        ↓
Learning Path Generation
        ↓
Final Recommendation
        ↓
Evaluation + Tracing
```

