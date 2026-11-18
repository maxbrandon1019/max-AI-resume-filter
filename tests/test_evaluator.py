"""Unit tests for JD Analyzer and Candidate Fit Scorer."""
from src.models.schemas import JDRequirements
from src.ingestion.sectionizer import ResumeSectionizer
from src.evaluator.jd_analyzer import JDAnalyzer
from src.evaluator.scorer import CandidateScorer
from src.retrieval.vector_store import HybridVectorStore

SAMPLE_JD = """
Senior AI & Machine Learning Engineer

Responsibilities:
- Build and scale RAG search pipelines.
- Deploy low-latency Python FastAPI services.

Requirements:
- 5+ years of experience in Python and ML.
- Required skills: Python, PyTorch, LangChain, FastAPI, Docker.
- Preferred skills: Kubernetes, AWS.
- Master's degree in CS or AI preferred.
"""

STRONG_RESUME = """
Maya Lin
maya@example.com
PROFESSIONAL SUMMARY
Lead AI Engineer with 6+ years experience in Python, PyTorch, LangChain, FastAPI, and Docker.
WORK EXPERIENCE
Lead AI Engineer | Tech
2018 - Present
Deployed RAG systems with FastAPI and PyTorch.
SKILLS
Python, PyTorch, LangChain, FastAPI, Docker, AWS
EDUCATION
Master of Science in Artificial Intelligence (2016 - 2018)
"""

WEAK_RESUME = """
Jordan Taylor
jordan@example.com
PROFESSIONAL SUMMARY
Product Manager with 4 years experience in Agile, Scrum, Product Roadmaps, and User Research.
WORK EXPERIENCE
Product Manager | Tech
2020 - Present
Managed SaaS roadmap.
SKILLS
Agile, Scrum, Product Strategy, SQL
EDUCATION
Bachelor of Arts in Economics
"""


def test_jd_analyzer():
    analyzer = JDAnalyzer()
    jd = analyzer.analyze(SAMPLE_JD)

    assert "Senior AI" in jd.job_title or "Engineer" in jd.job_title
    assert jd.min_years_experience >= 4.0
    assert len(jd.required_skills) > 0


def test_candidate_fit_scoring_strong_vs_weak():
    analyzer = JDAnalyzer()
    jd = analyzer.analyze(SAMPLE_JD)

    p_strong = ResumeSectionizer.sectionize(STRONG_RESUME, filename="maya.txt")
    p_weak = ResumeSectionizer.sectionize(WEAK_RESUME, filename="jordan.txt")

    store = HybridVectorStore()
    store.add_chunks(ResumeSectionizer.create_chunks(p_strong))
    store.add_chunks(ResumeSectionizer.create_chunks(p_weak))

    scorer = CandidateScorer(vector_store=store)

    eval_strong = scorer.evaluate(p_strong, jd)
    eval_weak = scorer.evaluate(p_weak, jd)

    # Strong candidate should score significantly higher
    assert eval_strong.overall_score > eval_weak.overall_score
    assert eval_strong.overall_score >= 70.0
    assert eval_strong.fit_tier in ("Strong Match", "Potential Match")
    assert eval_strong.matched_skills_count > eval_weak.matched_skills_count

    # Verify explainability artifacts
    assert len(eval_strong.strengths) > 0
    assert len(eval_strong.tailored_interview_questions) > 0
    assert len(eval_weak.gaps) > 0
