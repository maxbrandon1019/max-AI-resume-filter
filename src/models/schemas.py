"""Pydantic schemas for resumes, sections, evaluations, and RAG retrieval."""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ResumeSectionType(str, Enum):
    HEADER = "header"
    SUMMARY = "summary"
    EXPERIENCE = "work_experience"
    SKILLS = "skills"
    EDUCATION = "education"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    OTHER = "other"


class ResumeSection(BaseModel):
    """Represents a logical section within a resume."""
    section_type: ResumeSectionType = ResumeSectionType.OTHER
    title: str = "Section"
    content: str = ""
    page_number: int = 1
    token_count: Optional[int] = None


class CandidateProfile(BaseModel):
    """Complete structured profile extracted from a candidate's resume."""
    candidate_id: str
    filename: str
    name: str = "Unknown Candidate"
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    total_years_experience: Optional[float] = 0.0
    highest_education: Optional[str] = None
    top_skills: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    raw_text: str = ""
    sections: List[ResumeSection] = Field(default_factory=list)
    is_anonymized: bool = False
    layout_type: str = "single-column"
    extraction_engine: str = "standard"
    is_scanned: bool = False


class ChunkDocument(BaseModel):
    """Indexed document chunk for dense and sparse RAG retrieval."""
    chunk_id: str
    candidate_id: str
    candidate_name: str
    section_type: ResumeSectionType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    combined_score: Optional[float] = None


class JDRequirements(BaseModel):
    """Structured requirements decomposed from a Job Description."""
    job_title: str = "Target Position"
    department: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = 0.0
    preferred_education: Optional[str] = None
    key_responsibilities: List[str] = Field(default_factory=list)
    raw_jd_text: str = ""


class SkillMatch(BaseModel):
    """Individual skill matching assessment."""
    skill_name: str
    matched: bool
    is_required: bool = True
    candidate_evidence: Optional[str] = None


class FitEvaluation(BaseModel):
    """Comprehensive, explainable fit evaluation of a candidate against a JD."""
    candidate_id: str
    candidate_name: str
    overall_score: float = Field(ge=0.0, le=100.0)
    technical_score: float = Field(ge=0.0, le=100.0)
    experience_score: float = Field(ge=0.0, le=100.0)
    education_score: float = Field(ge=0.0, le=100.0)
    fit_tier: str = "Potential Match"  # "Strong Match" | "Potential Match" | "Weak Match"
    skills_matrix: List[SkillMatch] = Field(default_factory=list)
    matched_skills_count: int = 0
    total_required_skills_count: int = 0
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    fit_summary: str = ""
    tailored_interview_questions: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    """Source grounding for LLM answers."""
    candidate_id: str
    candidate_name: str
    section_type: str
    snippet: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Grounded response for conversational resume reader agent."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    candidate_ids_referenced: List[str] = Field(default_factory=list)
