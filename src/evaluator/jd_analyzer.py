"""Decomposes raw Job Descriptions into structured requirements."""
import re
from typing import Optional, List
from src.models.schemas import JDRequirements
from src.evaluator.llm_client import LLMClient


class JDAnalyzer:
    """Extracts mandatory skills, preferred skills, min years, and responsibilities from a Job Description."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def analyze(self, raw_jd: str) -> JDRequirements:
        """Parses raw JD text into structured JDRequirements object."""
        clean_jd = raw_jd.strip()
        if not clean_jd:
            return JDRequirements(job_title="General Role", raw_jd_text="")

        # 1. Try LLM Structured Decomposition
        if self.llm.is_live_llm_available:
            system_prompt = (
                "You are an expert technical recruiting specialist. Analyze the job description below and extract "
                "structured requirements: Job Title, Required Skills (mandatory hard skills), Preferred Skills (nice-to-haves), "
                "Minimum Years of Experience (numeric float), Preferred Education Degree, and Key Responsibilities."
            )
            parsed = self.llm.generate_structured(
                prompt=f"Job Description:\n{clean_jd}",
                system_prompt=system_prompt,
                schema=JDRequirements
            )
            if parsed:
                parsed.raw_jd_text = clean_jd
                return parsed

        # 2. Rule-Based Fallback Parser
        return self._heuristic_parse(clean_jd)

    def _heuristic_parse(self, jd_text: str) -> JDRequirements:
        """Rule-based extractor for job requirements."""
        lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
        
        # 1. Extract Job Title (usually line 1 or header)
        job_title = "Target Position"
        for line in lines[:3]:
            if len(line.split()) <= 6 and not any(kw in line.lower() for kw in ["about", "responsibilities", "requirements", "description"]):
                job_title = line.strip("# :*-")
                break

        # 2. Extract Min Years Experience
        min_years = 2.0
        exp_match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:-\s*\d+)?\s*years?(?:\s+of)?\s+experience", jd_text, re.IGNORECASE)
        if exp_match:
            try:
                min_years = float(exp_match.group(1))
            except ValueError:
                pass

        # 3. Detect Skills
        from src.ingestion.sectionizer import ResumeSectionizer
        all_skills = ResumeSectionizer._extract_top_skills(jd_text)
        
        # Divide into required (first 60%) and preferred (remaining)
        split_idx = max(1, int(len(all_skills) * 0.65))
        required = all_skills[:split_idx] if all_skills else ["General Technical Skills"]
        preferred = all_skills[split_idx:]

        # 4. Detect Preferred Education
        education = "Bachelor's Degree in Computer Science or related field"
        if "master" in jd_text.lower() or "m.s." in jd_text.lower():
            education = "Master's Degree Preferred"
        elif "phd" in jd_text.lower() or "ph.d" in jd_text.lower():
            education = "Ph.D. Preferred"

        # 5. Extract bullet point responsibilities
        responsibilities = []
        for line in lines:
            if line.startswith(("-", "*", "•")) and len(line) > 15:
                responsibilities.append(line.lstrip("-*• ").strip())
            if len(responsibilities) >= 5:
                break

        if not responsibilities:
            responsibilities = ["Design, develop, and deploy scalable systems", "Collaborate with cross-functional teams"]

        return JDRequirements(
            job_title=job_title,
            department="Engineering",
            required_skills=required,
            preferred_skills=preferred,
            min_years_experience=min_years,
            preferred_education=education,
            key_responsibilities=responsibilities,
            raw_jd_text=jd_text
        )
