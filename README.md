# RAG-Based Resume Filter & Reader

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](src/ui/app.py)
[![PyMuPDF Layout](https://img.shields.io/badge/PyMuPDF-Layout%20Aware-008080.svg)](src/ingestion/parser.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end **Retrieval-Augmented Generation (RAG)** candidate intelligence system designed to ingest, parse, index, filter, rank, and interactively query candidate resumes against Job Descriptions (JDs) with explainable AI scoring, visual radar fit analytics, head-to-head candidate comparisons, and conversational resume analysis.

---

## 🌟 Key Features

- **Multi-Column Layout-Aware Resume Parsing**:
  - **PyMuPDF (`fitz`) Layout Engine**: Dynamically detects vertical column gutters, separates page headers/footers, and sequences multi-column text top-to-bottom per column (eliminating horizontal column mixing).
  - **NFKC Ligature Normalization**: Resolves typographical ligatures (`\ufb01` $\to$ `fi`, `\ufb02` $\to$ `fl`), de-hyphenates line breaks, and standardizes custom bullet glyphs.
  - **Scanned Document Recovery**: Automatically flags image-only PDFs ($<35$ words) and triggers Gemini Multimodal Vision OCR.
  - Multi-format extraction for **PDF**, **DOCX**, and **TXT** files.
- **Explainable Fit Scoring & Visual Radar Analytics**:
  - Automatically deconstructs unstructured Job Descriptions into mandatory and preferred requirements.
  - Calculates transparent **Overall (0–100)**, **Technical**, **Experience**, and **Education** scores.
  - **Dependency-Free SVG Radar Chart**: Renders responsive visual spider charts mapping candidate proficiency across 5 key dimensions.
- **Head-to-Head Candidate Comparison Matrix**:
  - Compare any two contenders side-by-side with overlapping radar graphs.
  - **Skill Differential Breakdown**: Identifies shared capabilities, Candidate A exclusive strengths, Candidate B exclusive strengths, and mutual missing skills.
  - Side-by-side qualitative strengths and risk factor breakdown with an automated hiring trade-off recommendation.
- **Dynamic Leaderboard Filtering & Sorting**:
  - Filter candidate pools by minimum fit score slider, fit tiers (`Strong Fit`, `Moderate Fit`, `Weak Fit`), and live text search across candidate names, degrees, and skills.
- **Bias-Free Screening (PII Masking)**:
  - 1-click anonymization redacting names, contact details, personal links, and locations to ensure objective evaluation.
- **Hybrid Dense + Sparse (BM25) Retrieval**:
  - Combines dense semantic cosine similarity with BM25 Okapi sparse keyword search and Cross-Encoder re-ranking.
- **Conversational Resume Reader Agent**:
  - Natural language Q&A across the entire talent pool or scoped to individual candidates with source citations and section attribution.
- **Executive Report Generation**:
  - Print-ready executive HTML briefs (`@media print`), Markdown dossiers, and CSV leaderboard exports.

---

## 🏗️ System Architecture

```
                                 [ Resume Files: PDF / DOCX / TXT ]
                                                 │
                                                 ▼
                                     [ Document Parser ]
                                                 │
                                                 ▼
                                   [ Section-Aware Chunking ]
                                 (Exp, Skills, Edu, Projects)
                                                 │
                                  ┌──────────────┴──────────────┐
                                  ▼                             ▼
                        [ Dense Embeddings ]            [ Sparse BM25 Index ]
                        (Gemini / OpenAI / Local)       (Keyword Term Freq)
                                  │                             │
                                  └──────────────┬──────────────┘
                                                 ▼
                                      [( Hybrid Vector Store )]
                                                 │
   [ Job Description / Recruiter Query ] ────────┼────────────────────────┐
                                                 ▼                        ▼
                                     [ Hybrid Rank Retrieval ]   [ Conversational Reader ]
                                                 │                        │
                                                 ▼                        ▼
                                      [ Cross-Encoder Re-rank ]   [ Grounded Answer + Citations ]
                                                 │
                                                 ▼
                                    [ Explainable Fit Scorer ]
                                   (0-100 Score, Gaps, Strengths)
                                                 │
                                                 ▼
                                   [ Recruiter UI Dashboard ]
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone or open the repository, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Configure your preferred LLM provider (`gemini`, `openai`, or `mock` for local zero-API mode):

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

> **Note**: The system includes a built-in deterministic offline engine. If no API keys are provided, it operates seamlessly in local fallback mode!

### 3. Launch the Recruiter Dashboard

```bash
streamlit run src/ui/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

The application is container-ready. Launch the complete system with Docker Compose:

```bash
# Build and run containerized dashboard
docker compose up --build
```

Then navigate to `http://localhost:8501`.

To stop the container:
```bash
docker compose down
```

---

## 🧪 Running Subsystem Tests

Run the built-in test suite to verify ingestion, hybrid retrieval, candidate scoring, and chat agents:

```bash
python run_tests.py
```

Or using `pytest`:

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```
Max_RAG_Resume_Project/
├── Dockerfile                     # Multi-stage production container definition
├── docker-compose.yml             # Single-command local container orchestration
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Production-grade git exclusions (prevents secret leaks)
├── configs/
│   └── app_config.yaml            # Weights, thresholds, and model parameters
├── data/
│   ├── sample_resumes/            # Pre-bundled diverse test resumes
│   └── sample_jds/                # Pre-bundled sample job descriptions
├── src/
│   ├── models/
│   │   └── schemas.py             # Pydantic data models & evaluation schemas
│   ├── ingestion/
│   │   ├── parser.py              # PyMuPDF column-aware extractor & ligature cleaner
│   │   ├── sectionizer.py         # Section segmentation & entity extraction
│   │   └── pii_masker.py          # Real-time PII anonymization engine
│   ├── retrieval/
│   │   ├── embeddings.py          # Gemini, OpenAI & Local feature embeddings
│   │   ├── vector_store.py        # Hybrid Vector + BM25 store with metadata filtering
│   │   └── reranker.py            # Re-ranking module
│   ├── evaluator/
│   │   ├── llm_client.py          # Multi-provider LLM & structured JSON generator
│   │   ├── jd_analyzer.py         # Job Description deconstructor
│   │   ├── scorer.py              # Explainable fit evaluation engine
│   │   └── report_generator.py    # Executive HTML brief, Markdown & CSV exports
│   ├── reader/
│   │   └── chat_agent.py          # Conversational RAG agent with citations
│   └── ui/
│       ├── app.py                 # Streamlit recruiter dashboard
│       └── charts.py              # Pure SVG radar & comparison chart engine
├── tests/
│   ├── test_parser.py
│   ├── test_retrieval.py
│   ├── test_evaluator.py
│   ├── test_report_generator.py
│   └── test_chat.py
├── run_tests.py                   # 23-step verification test suite
├── requirements.txt
└── README.md
```

---

## 💡 Programmatic Usage Example

```python
from src.ingestion.sectionizer import ResumeSectionizer
from src.retrieval.vector_store import HybridVectorStore
from src.evaluator.jd_analyzer import JDAnalyzer
from src.evaluator.scorer import CandidateScorer
from src.reader.chat_agent import ResumeChatAgent

# 1. Ingest Resume
with open("data/sample_resumes/maya_lin_ai_ml_engineer.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

candidate = ResumeSectionizer.sectionize(raw_text, filename="maya_lin.txt")
chunks = ResumeSectionizer.create_chunks(candidate)

# 2. Index in Hybrid Vector Store
store = HybridVectorStore()
store.add_chunks(chunks)

# 3. Analyze JD and Score Candidate
with open("data/sample_jds/senior_ai_python_engineer.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

analyzer = JDAnalyzer()
jd = analyzer.analyze(jd_text)

scorer = CandidateScorer(vector_store=store)
evaluation = scorer.evaluate(candidate, jd)

print(f"Overall Fit Score: {evaluation.overall_score}/100 ({evaluation.fit_tier})")
print(f"Technical Match: {evaluation.technical_score}%")
print(f"Strengths: {evaluation.strengths}")
print(f"Gaps: {evaluation.gaps}")

# 4. Interactive Conversational Reader
chat_agent = ResumeChatAgent(vector_store=store, candidates={candidate.candidate_id: candidate})
response = chat_agent.chat("What RAG frameworks and vector databases has Maya worked with?")
print(f"\nAnswer: {response.answer}")
```
