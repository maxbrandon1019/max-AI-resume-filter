"""Unit tests for Executive Report Generator (HTML, Markdown, CSV)."""
from src.models.schemas import CandidateProfile, FitEvaluation, SkillMatch, JDRequirements
from src.evaluator.report_generator import ReportGenerator


def _create_mock_eval_and_profile():
    profile = CandidateProfile(
        candidate_id="cand_12345",
        filename="maya_lin.pdf",
        name="Maya Lin",
        email="maya@example.com",
        phone="(555) 019-2834",
        total_years_experience=6.5,
        highest_education="Master's Degree",
        top_skills=["Python", "PyTorch", "LangChain", "FastAPI", "Docker"],
        raw_text="Maya Lin Lead AI Engineer with PyTorch, FastAPI, and Docker experience."
    )

    jd = JDRequirements(
        job_title="Senior AI Engineer",
        required_skills=["Python", "PyTorch", "LangChain", "FastAPI"],
        preferred_skills=["Kubernetes", "AWS"],
        min_years_experience=5.0,
        preferred_education="Master's Degree Preferred"
    )

    evaluation = FitEvaluation(
        candidate_id="cand_12345",
        candidate_name="Maya Lin",
        overall_score=88.5,
        technical_score=92.0,
        experience_score=85.0,
        education_score=90.0,
        fit_tier="Strong Match",
        matched_skills_count=4,
        total_required_skills_count=4,
        skills_matrix=[
            SkillMatch(skill_name="Python", matched=True, is_required=True, candidate_evidence="Expert Python engineer"),
            SkillMatch(skill_name="PyTorch", matched=True, is_required=True, candidate_evidence="Built deep learning models with PyTorch"),
            SkillMatch(skill_name="Kubernetes", matched=False, is_required=False)
        ],
        strengths=["Deep expertise in PyTorch and RAG architectures", "Meets 6+ years seniority bar"],
        gaps=["No explicit mention of Kubernetes in production"],
        fit_summary="Maya Lin is a Strong Match (Score: 88.5/100) with deep domain alignment.",
        tailored_interview_questions=[
            "Can you describe your largest production RAG pipeline architecture?",
            "How do you approach container orchestration when Kubernetes is not available?"
        ]
    )

    return profile, jd, evaluation


def test_html_report_generation():
    profile, jd, evaluation = _create_mock_eval_and_profile()
    html = ReportGenerator.generate_executive_html(profile, evaluation, jd)

    assert "<!DOCTYPE html>" in html
    assert "Maya Lin" in html
    assert "Senior AI Engineer" in html
    assert "88.5" in html
    assert "Strong Match" in html
    assert "@media print" in html
    assert "PyTorch" in html
    assert "Targeted Interview Assessment Rubric" in html
    assert "Q1" in html


def test_markdown_dossier_generation():
    profile, jd, evaluation = _create_mock_eval_and_profile()
    md = ReportGenerator.generate_markdown_dossier(profile, evaluation, jd)

    assert "# Candidate Evaluation Dossier: Maya Lin" in md
    assert "**Target Role:** Senior AI Engineer" in md
    assert "`88.5 / 100`" in md
    assert "## 🧩 Detailed Skill Matrix" in md
    assert "PyTorch" in md
    assert "## 🎯 Tailored Interview Questions" in md


def test_leaderboard_csv_generation():
    profile, jd, evaluation = _create_mock_eval_and_profile()
    csv_str = ReportGenerator.generate_leaderboard_csv([evaluation], {profile.candidate_id: profile})

    lines = [line.strip() for line in csv_str.strip().splitlines()]
    assert len(lines) == 2  # Header + 1 candidate
    header = lines[0]
    assert "Candidate Name" in header
    assert "Overall Score" in header
    assert "Fit Tier" in header

    row = lines[1]
    assert "Maya Lin" in row
    assert "Strong Match" in row
    assert "88.5" in row
