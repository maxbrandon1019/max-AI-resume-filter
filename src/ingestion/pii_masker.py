"""PII (Personally Identifiable Information) Redaction & Masking module."""
import re
from typing import Tuple
from src.models.schemas import CandidateProfile, ResumeSection


class PIIMasker:
    """Masks personal contact info and candidate names for unbiased resume screening."""

    @classmethod
    def mask_profile(cls, profile: CandidateProfile) -> CandidateProfile:
        """Returns an anonymized copy of the candidate profile."""
        anonymized = profile.model_copy(deep=True)
        orig_name = anonymized.name
        masked_id = f"Candidate-{anonymized.candidate_id[:6].upper()}"

        anonymized.name = masked_id
        anonymized.email = "[REDACTED_EMAIL]" if anonymized.email else None
        anonymized.phone = "[REDACTED_PHONE]" if anonymized.phone else None
        anonymized.location = "[REDACTED_LOCATION]" if anonymized.location else None
        anonymized.links = ["[REDACTED_LINK]"] * len(anonymized.links)
        anonymized.is_anonymized = True

        # Redact locations in structured work experiences
        if anonymized.work_experiences:
            for exp in anonymized.work_experiences:
                if exp.location:
                    exp.location = "[REDACTED_LOCATION]"

        # Mask inside raw text and sections
        if orig_name and orig_name != "Unknown Candidate":
            name_pattern = re.compile(re.escape(orig_name), re.IGNORECASE)
            anonymized.raw_text = name_pattern.sub(masked_id, anonymized.raw_text)
            if anonymized.summary:
                anonymized.summary = name_pattern.sub(masked_id, anonymized.summary)

            for sec in anonymized.sections:
                sec.content = name_pattern.sub(masked_id, sec.content)

        # Mask emails and phones in sections
        for sec in anonymized.sections:
            sec.content = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", sec.content)
            sec.content = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[REDACTED_PHONE]", sec.content)

        return anonymized
