"""Candidate Fit Scorer and Explainable Evaluation Engine."""
import re
from typing import List, Optional
from src.models.schemas import (
    CandidateProfile,
    JDRequirements,
    FitEvaluation,
    SkillMatch,
    ChunkDocument
)
from src.evaluator.llm_client import LLMClient
from src.retrieval.vector_store import HybridVectorStore


class CandidateScorer:
    """Evaluates candidate resumes against job descriptions with explainable scoring."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        vector_store: Optional[HybridVectorStore] = None
    ):
        self.llm = llm_client or LLMClient()
        self.vector_store = vector_store

    def evaluate(self, candidate: CandidateProfile, jd: JDRequirements) -> FitEvaluation:
        """
        Performs full explainable fit evaluation of a candidate against a job description.
        """
        # 1. Retrieve RAG Evidence Chunks for this candidate
        evidence_chunks: List[ChunkDocument] = []
        if self.vector_store:
            query = f"{jd.job_title} {' '.join(jd.required_skills[:5])} {' '.join(jd.key_responsibilities[:2])}"
            evidence_chunks = self.vector_store.search_for_candidate(
                candidate_id=candidate.candidate_id,
                query=query,
                top_k=4
            )

        # 2. Try LLM Deep Evaluation if available
        if self.llm.is_live_llm_available:
            try:
                eval_result = self._evaluate_with_llm(candidate, jd, evidence_chunks)
                if eval_result:
                    return eval_result
            except Exception:
                pass

        # 3. Deterministic Rule-Based Evaluation Engine Fallback
        return self._evaluate_heuristic(candidate, jd, evidence_chunks)

    def _evaluate_with_llm(
        self,
        candidate: CandidateProfile,
        jd: JDRequirements,
        evidence_chunks: List[ChunkDocument]
    ) -> Optional[FitEvaluation]:
        """Uses LLM to generate deep qualitative fit evaluation."""
        system_prompt = (
            "You are a Senior Principal Technical Recruiter and Engineering Hiring Manager. "
            "Evaluate the candidate resume against the Job Description. Be objective, realistic, and rigorous. "
            "Output a structured JSON response matching the FitEvaluation schema."
        )

        context_evidence = "\n\n".join([f"[{c.metadata.get('section_type', 'Section')}]: {c.content}" for c in evidence_chunks])
        if not context_evidence:
            context_evidence = candidate.raw_text[:3000]

        prompt = (
            f"=== JOB DESCRIPTION ===\n"
            f"Title: {jd.job_title}\n"
            f"Min Years: {jd.min_years_experience}\n"
            f"Required Skills: {', '.join(jd.required_skills)}\n"
            f"Preferred Skills: {', '.join(jd.preferred_skills)}\n"
            f"Responsibilities:\n" + "\n".join(f"- {r}" for r in jd.key_responsibilities[:4]) + "\n\n"
            f"=== CANDIDATE RESUME EVIDENCE ===\n"
            f"Name: {candidate.name}\n"
            f"Detected Experience: {candidate.total_years_experience} years\n"
            f"Education: {candidate.highest_education}\n"
            f"Extracted Skills: {', '.join(candidate.top_skills)}\n\n"
            f"Key Resume Sections:\n{context_evidence}\n"
        )

        evaluation = self.llm.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            schema=FitEvaluation
        )

        if evaluation:
            evaluation.candidate_id = candidate.candidate_id
            evaluation.candidate_name = candidate.name
            return evaluation

        return None

    def _evaluate_heuristic(
        self,
        candidate: CandidateProfile,
        jd: JDRequirements,
        evidence_chunks: List[ChunkDocument]
    ) -> FitEvaluation:
        """Deterministic, transparent scoring engine."""
        cand_text_lower = candidate.raw_text.lower()
        cand_skills_lower = {s.lower() for s in candidate.top_skills}

        # 1. Skill Matrix Evaluation
        skills_matrix: List[SkillMatch] = []
        matched_req_count = 0
        total_req_count = len(jd.required_skills) if jd.required_skills else 1

        for skill in jd.required_skills:
            s_lower = skill.lower()
            matched = (s_lower in cand_skills_lower) or (re.search(rf"\b{re.escape(s_lower)}\b", cand_text_lower) is not None)
            evidence = self._find_evidence_snippet(candidate.raw_text, skill) if matched else None
            if matched:
                matched_req_count += 1
            skills_matrix.append(SkillMatch(
                skill_name=skill,
                matched=matched,
                is_required=True,
                candidate_evidence=evidence
            ))

        for skill in jd.preferred_skills:
            s_lower = skill.lower()
            matched = (s_lower in cand_skills_lower) or (re.search(rf"\b{re.escape(s_lower)}\b", cand_text_lower) is not None)
            evidence = self._find_evidence_snippet(candidate.raw_text, skill) if matched else None
            skills_matrix.append(SkillMatch(
                skill_name=skill,
                matched=matched,
                is_required=False,
                candidate_evidence=evidence
            ))

        # 2. Score Calculations
        # Technical Score
        tech_ratio = matched_req_count / max(1, total_req_count)
        technical_score = min(100.0, max(15.0, round(tech_ratio * 90.0 + (10.0 if any(sm.matched for sm in skills_matrix if not sm.is_required) else 0.0), 1)))

        # Experience Score
        req_exp = jd.min_years_experience or 1.0
        cand_exp = candidate.total_years_experience or 0.0
        if cand_exp >= req_exp:
            experience_score = min(100.0, 85.0 + min(15.0, (cand_exp - req_exp) * 3.0))
        else:
            exp_ratio = cand_exp / max(1.0, req_exp)
            experience_score = max(20.0, round(exp_ratio * 80.0, 1))

        # Education Score
        education_score = 80.0
        cand_edu = (candidate.highest_education or "").lower()
        if "ph" in cand_edu or "doctor" in cand_edu:
            education_score = 95.0
        elif "master" in cand_edu:
            education_score = 90.0
        elif "bachelor" in cand_edu:
            education_score = 80.0
        elif "associate" in cand_edu or "diploma" in cand_edu:
            education_score = 65.0

        # Overall Score (Weighted)
        overall_score = round(
            0.45 * technical_score +
            0.35 * experience_score +
            0.20 * education_score,
            1
        )

        # Fit Tier
        if overall_score >= 75.0:
            fit_tier = "Strong Match"
        elif overall_score >= 55.0:
            fit_tier = "Potential Match"
        else:
            fit_tier = "Weak Match"

        # 3. Strengths & Gaps Generation
        strengths = []
        gaps = []

        matched_skills = [sm.skill_name for sm in skills_matrix if sm.matched]
        missing_skills = [sm.skill_name for sm in skills_matrix if not sm.matched and sm.is_required]

        if matched_skills:
            strengths.append(f"Demonstrated proficiency in core required technologies: {', '.join(matched_skills[:4])}.")
        if cand_exp >= req_exp:
            strengths.append(f"Meets seniority requirement with ~{int(cand_exp)} years of detected relevant industry experience (JD requires {int(req_exp)}+ years).")
        if candidate.highest_education:
            strengths.append(f"Holds relevant academic background: {candidate.highest_education}.")

        if missing_skills:
            gaps.append(f"Missing explicit mentions for key required skills: {', '.join(missing_skills[:3])}.")
        if cand_exp < req_exp:
            gaps.append(f"Experience level (~{cand_exp:.1f} yrs) is below requested {req_exp} years minimum.")

        if not strengths:
            strengths.append("Found general technical background aligned with software engineering.")
        if not gaps:
            gaps.append("No critical technical or experience deficiencies identified against requirements.")

        # 4. Tailored Interview Questions
        questions = []
        if missing_skills:
            questions.append(f"How would you approach ramping up on {missing_skills[0]} given your prior background?")
        if matched_skills:
            questions.append(f"Can you walk us through an end-to-end architecture you designed utilizing {matched_skills[0]}?")
        questions.append(f"What was the most challenging technical project you delivered in the past 2 years, and how did you measure its success?")

        fit_summary = (
            f"{candidate.name} is assessed as a {fit_tier} (Score: {overall_score}/100). "
            f"Matched {matched_req_count}/{total_req_count} mandatory skills with {cand_exp:.1f} years of relevant experience."
        )

        return FitEvaluation(
            candidate_id=candidate.candidate_id,
            candidate_name=candidate.name,
            overall_score=overall_score,
            technical_score=technical_score,
            experience_score=experience_score,
            education_score=education_score,
            fit_tier=fit_tier,
            skills_matrix=skills_matrix,
            matched_skills_count=matched_req_count,
            total_required_skills_count=total_req_count,
            strengths=strengths,
            gaps=gaps,
            red_flags=[],
            fit_summary=fit_summary,
            tailored_interview_questions=questions
        )

    def _find_evidence_snippet(self, text: str, term: str) -> Optional[str]:
        """Finds a 1-2 sentence context snippet containing the search term."""
        pattern = re.compile(rf"([^.\n]*?\b{re.escape(term)}\b[^.\n]*\.)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return f"Mentioned in resume ({term})"
