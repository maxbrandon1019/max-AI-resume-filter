"""Section-aware chunking and entity extraction from raw resume text."""
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from src.models.schemas import (
    CandidateProfile,
    ResumeSection,
    ResumeSectionType,
    ChunkDocument
)


class ResumeSectionizer:
    """Splits raw resume text into logical sections and extracts structured metadata with high precision."""

    # Expanded Section Header Patterns
    SECTION_PATTERNS = [
        (
            ResumeSectionType.SUMMARY,
            re.compile(r"^(?:professional\s+|executive\s+|career\s+)?(?:summary|profile|objective|about\s+me|overview|biography)\b", re.IGNORECASE)
        ),
        (
            ResumeSectionType.EXPERIENCE,
            re.compile(r"^(?:work\s+|professional\s+|employment\s+|career\s+|relevant\s+|industry\s+)?(?:experience|history|background|employment|positions\s+held)\b", re.IGNORECASE)
        ),
        (
            ResumeSectionType.SKILLS,
            re.compile(r"^(?:technical\s+|core\s+|key\s+|specialized\s+)?(?:skills|technologies|competencies|proficiencies|tech\s+stack|tools|expertise|programming\s+languages|tooling)\b", re.IGNORECASE)
        ),
        (
            ResumeSectionType.EDUCATION,
            re.compile(r"^(?:academic\s+|educational\s+)?(?:education|background|qualifications|degrees|academics|university)\b", re.IGNORECASE)
        ),
        (
            ResumeSectionType.PROJECTS,
            re.compile(r"^(?:personal\s+|key\s+|selected\s+|technical\s+|featured\s+)?(?:projects|portfolio|open\s+source|initiatives)\b", re.IGNORECASE)
        ),
        (
            ResumeSectionType.CERTIFICATIONS,
            re.compile(r"^(?:professional\s+)?(?:certifications|licenses|courses|certificates|credentials|training|accreditations)\b", re.IGNORECASE)
        )
    ]

    # Email & Phone & URL regexes
    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_REGEX = re.compile(r"(?:\+?\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,9}")
    URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s,]+|linkedin\.com/in/[^\s,]+|github\.com/[^\s,]+")

    # Modern Tech Skills Taxonomy
    COMMON_SKILLS = [
        # AI & Machine Learning & RAG
        "python", "rag", "langchain", "llamaindex", "chromadb", "qdrant", "pinecone", "weaviate", "faiss", "milvus",
        "pytorch", "tensorflow", "keras", "scikit-learn", "huggingface", "transformers", "nlp", "computer vision",
        "deep learning", "machine learning", "mlops", "mlflow", "triton", "ray", "onnx", "pandas", "numpy",
        "llm", "large language models", "generative ai", "fine-tuning", "lora", "qlora", "prompt engineering",
        "vllm", "ollama", "openai", "gemini", "claude",
        # Languages & Core Backend
        "javascript", "typescript", "java", "c++", "c#", "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala",
        "sql", "fastapi", "django", "flask", "spring boot", "node.js", "express", "nestjs", "grpc", "rest api", "graphql", "celery",
        # Frontend
        "react", "react.js", "angular", "vue", "vue.js", "next.js", "svelte", "tailwind", "tailwind css", "redux", "html5", "css3", "vite", "webpack",
        # Cloud & DevOps
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform", "ansible",
        "ci/cd", "jenkins", "github actions", "gitlab ci", "argocd", "helm", "linux", "bash", "prometheus", "grafana", "datadog", "elk", "istio",
        # Databases & Messaging & Big Data
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb", "sqlite", "supabase", "neo4j",
        "kafka", "apache kafka", "rabbitmq", "snowflake", "bigquery", "spark", "apache spark", "airflow", "dbt",
        # Architecture & Practices
        "microservices", "distributed systems", "event-driven architecture", "system design", "git", "agile", "scrum", "pytest", "unit testing"
    ]

    # Skill Synonym Map (maps alias/variant -> canonical name)
    SKILL_SYNONYMS = {
        "k8s": "Kubernetes",
        "golang": "Go",
        "postgres": "PostgreSQL",
        "react.js": "React",
        "reactjs": "React",
        "vue.js": "Vue",
        "vuejs": "Vue",
        "nextjs": "Next.js",
        "next.js": "Next.js",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "amazon web services": "AWS",
        "google cloud": "GCP",
        "google cloud platform": "GCP",
        "llms": "LLM",
        "large language models": "LLM",
        "generative ai": "Generative AI",
        "genai": "Generative AI",
        "deep learning": "Deep Learning",
        "machine learning": "Machine Learning",
        "restful api": "REST API",
        "restful apis": "REST API",
        "tailwind css": "Tailwind",
        "tailwindcss": "Tailwind",
        "apache kafka": "Kafka",
        "apache spark": "Spark"
    }

    MONTHS_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    @classmethod
    def sectionize(
        cls,
        raw_text: str,
        filename: str = "resume.pdf",
        layout_type: str = "single-column",
        extraction_engine: str = "standard",
        is_scanned: bool = False
    ) -> CandidateProfile:
        """Parses raw text into structured CandidateProfile with recognized sections."""
        clean_text = raw_text.strip()
        candidate_id = hashlib.md5(f"{filename}_{clean_text[:100]}".encode("utf-8")).hexdigest()[:12]

        raw_lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        lines = cls._preprocess_lines(raw_lines)

        # 1. Extract contact & header info
        email = cls._extract_email(clean_text)
        phone = cls._extract_phone(clean_text)
        links = cls._extract_links(clean_text)
        name = cls._extract_name(lines, email)

        # 2. Segment text into sections
        sections = cls._split_into_sections(lines)

        # 3. Extract high-level qualifications
        total_exp = cls._calculate_years_experience(clean_text, sections)
        education = cls._detect_highest_education(clean_text)
        top_skills = cls._extract_top_skills(clean_text)
        summary = cls._extract_summary(sections)

        return CandidateProfile(
            candidate_id=candidate_id,
            filename=filename,
            name=name,
            email=email,
            phone=phone,
            location=cls._extract_location(clean_text),
            links=links,
            total_years_experience=total_exp,
            highest_education=education,
            top_skills=top_skills,
            summary=summary,
            raw_text=clean_text,
            sections=sections,
            layout_type=layout_type,
            extraction_engine=extraction_engine,
            is_scanned=is_scanned
        )

    @classmethod
    def create_chunks(cls, profile: CandidateProfile) -> List[ChunkDocument]:
        """Converts CandidateProfile sections into indexable ChunkDocuments for RAG."""
        chunks: List[ChunkDocument] = []

        for idx, sec in enumerate(profile.sections):
            chunk_id = f"{profile.candidate_id}_sec_{idx}_{sec.section_type.value}"
            metadata = {
                "candidate_id": profile.candidate_id,
                "candidate_name": profile.name,
                "section_type": sec.section_type.value,
                "filename": profile.filename,
                "years_experience": profile.total_years_experience,
                "education": profile.highest_education,
                "top_skills": ", ".join(profile.top_skills[:10])
            }
            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    candidate_id=profile.candidate_id,
                    candidate_name=profile.name,
                    section_type=sec.section_type,
                    content=f"Candidate: {profile.name}\nSection: {sec.title.upper()}\n\n{sec.content}",
                    metadata=metadata
                )
            )

        # Overview summary chunk for broad talent pool queries
        summary_chunk_id = f"{profile.candidate_id}_overview"
        overview_text = (
            f"Candidate Name: {profile.name}\n"
            f"Experience: {profile.total_years_experience} years\n"
            f"Education: {profile.highest_education}\n"
            f"Top Skills: {', '.join(profile.top_skills)}\n"
            f"Summary: {profile.summary or 'N/A'}\n"
        )
        chunks.append(
            ChunkDocument(
                chunk_id=summary_chunk_id,
                candidate_id=profile.candidate_id,
                candidate_name=profile.name,
                section_type=ResumeSectionType.SUMMARY,
                content=overview_text,
                metadata={
                    "candidate_id": profile.candidate_id,
                    "candidate_name": profile.name,
                    "section_type": "profile_overview",
                    "filename": profile.filename
                }
            )
        )

        return chunks

    @classmethod
    def _preprocess_lines(cls, raw_lines: List[str]) -> List[str]:
        """
        Preprocesses lines to separate headings from content when on the same line
        (e.g., 'SKILLS: Python, Go, Docker' -> ['SKILLS', 'Python, Go, Docker'])
        and strips markdown markers (#, ##, **).
        """
        processed: List[str] = []
        for line in raw_lines:
            clean = line.strip()
            # Strip markdown heading characters (#, ##) and bold wrappers
            clean = re.sub(r"^#{1,4}\s*", "", clean)
            clean = re.sub(r"^\*\*(.*?)\*\*$", r"\1", clean).strip()

            if not clean:
                continue

            # Check if line has a section heading prefix followed by separator
            m = re.match(r"^([A-Za-z\s&/]{3,35})\s*[:\-–—]\s*(.+)$", clean)
            if m:
                prefix = m.group(1).strip()
                rest = m.group(2).strip()
                matched = False
                for sec_type, pat in cls.SECTION_PATTERNS:
                    if pat.search(prefix):
                        processed.append(prefix)
                        processed.append(rest)
                        matched = True
                        break
                if not matched:
                    processed.append(clean)
            else:
                processed.append(clean)

        return processed

    @classmethod
    def _split_into_sections(cls, lines: List[str]) -> List[ResumeSection]:
        """Splits raw lines into recognized resume sections based on header patterns."""
        sections: List[ResumeSection] = []
        current_type = ResumeSectionType.HEADER
        current_title = "Header & Contact"
        current_lines: List[str] = []

        def flush_section():
            nonlocal current_lines, current_type, current_title
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append(
                        ResumeSection(
                            section_type=current_type,
                            title=current_title,
                            content=content,
                            token_count=len(content.split())
                        )
                    )
            current_lines = []

        for line in lines:
            matched_type = None
            clean_line = line.strip(" :-#=")

            # Section headings are typically 1 to 5 words
            if len(clean_line.split()) <= 6:
                for sec_type, pattern in cls.SECTION_PATTERNS:
                    if pattern.search(clean_line):
                        matched_type = sec_type
                        break

            if matched_type:
                flush_section()
                current_type = matched_type
                current_title = clean_line.title()
            else:
                current_lines.append(line)

        flush_section()
        return sections

    @classmethod
    def _extract_name(cls, lines: List[str], email: Optional[str]) -> str:
        """Robust candidate name extraction handling multi-part lines and contact banners."""
        if not lines:
            return "Unknown Candidate"

        # Check top 4 lines
        for line in lines[:4]:
            # Split by common header delimiters (pipe, bullet, tab, comma)
            segments = [s.strip() for s in re.split(r"[|•·,\t]", line) if s.strip()]
            for seg in segments:
                clean_seg = re.sub(r"[^\w\s\.\-]", "", seg).strip()
                words = clean_seg.split()
                if 2 <= len(words) <= 4 and all(w.replace(".", "").replace("-", "").isalpha() for w in words):
                    if not cls.EMAIL_REGEX.search(seg) and not cls.PHONE_REGEX.search(seg):
                        lower = clean_seg.lower()
                        # Exclude common job title or resume terms
                        if not any(kw in lower for kw in ["resume", "curriculum", "vitae", "profile", "engineer", "developer", "manager", "architect", "designer", "street", "drive", "lane", "page"]):
                            return clean_seg.title()

        # Fallback to email username if cleanly formatted
        if email:
            user_part = email.split("@")[0]
            parts = re.split(r"[._-]", user_part)
            if len(parts) >= 2 and all(p.isalpha() for p in parts):
                return " ".join(parts).title()

        # Final fallback: first line text
        first_line = lines[0].split("|")[0].split("•")[0].strip()
        return first_line[:40].strip().title()

    @classmethod
    def _extract_email(cls, text: str) -> Optional[str]:
        match = cls.EMAIL_REGEX.search(text)
        return match.group(0) if match else None

    @classmethod
    def _extract_phone(cls, text: str) -> Optional[str]:
        match = cls.PHONE_REGEX.search(text)
        return match.group(0).strip() if match else None

    @classmethod
    def _extract_links(cls, text: str) -> List[str]:
        matches = cls.URL_REGEX.findall(text)
        return list(dict.fromkeys(matches))[:5]

    @classmethod
    def _extract_location(cls, text: str) -> Optional[str]:
        loc_match = re.search(r"\b([A-Z][a-zA-Z\s]+,\s*[A-Z]{2,}(?:\s*,\s*[A-Z][a-zA-Z]+)?)\b", text[:600])
        return loc_match.group(1).strip() if loc_match else None

    @classmethod
    def _parse_date_point(cls, s: str, default_year: int) -> Optional[float]:
        """Parses a date string (e.g. 'Jan 2021', '03/2019', '2020') into a decimal year."""
        s = s.strip().lower()
        if s in ("present", "current", "now", "today"):
            return default_year + datetime.now().month / 12.0

        # Try Month Name + Year ('Jan 2020' or 'January 2020')
        m = re.search(r"([a-z]{3,9})[\s,]+(19\d\d|20\d\d)", s)
        if m:
            month_str = m.group(1)[:3]
            month = cls.MONTHS_MAP.get(month_str, 6)
            year = int(m.group(2))
            return year + (month - 1) / 12.0

        # Try MM/YYYY or MM.YYYY ('03/2020' or '3.2020')
        m2 = re.search(r"(\d{1,2})[\/\.](19\d\d|20\d\d)", s)
        if m2:
            month = min(12, max(1, int(m2.group(1))))
            year = int(m2.group(2))
            return year + (month - 1) / 12.0

        # Try plain year '2020'
        m3 = re.search(r"\b(19\d\d|20\d\d)\b", s)
        if m3:
            return float(m3.group(1))

        return None

    @classmethod
    def _calculate_years_experience(cls, text: str, sections: List[ResumeSection]) -> float:
        """Estimates total years of experience with month-year precision and interval merging."""
        current_year = datetime.now().year

        # 1. Check explicit mentions like '8+ years of experience'
        exp_match = re.search(r"(\d+(?:\.\d+)?)\+?\s*years?(?:\s+of)?\s+experience", text, re.IGNORECASE)
        if exp_match:
            try:
                val = float(exp_match.group(1))
                if 0.5 <= val <= 40.0:
                    return val
            except ValueError:
                pass

        # 2. Extract date spans across experience sections or full text
        target_text = ""
        for sec in sections:
            if sec.section_type in {ResumeSectionType.EXPERIENCE, ResumeSectionType.SUMMARY}:
                target_text += "\n" + sec.content
        if not target_text.strip():
            target_text = text

        date_span_pattern = re.compile(
            r"((?:[a-zA-Z]{3,9}\s+)?(?:\d{1,2}[\/\.])?(?:19\d\d|20\d\d))\s*(?:-|–|—|to)\s*((?:[a-zA-Z]{3,9}\s+)?(?:\d{1,2}[\/\.])?(?:19\d\d|20\d\d)|present|current|now)",
            re.IGNORECASE
        )

        intervals: List[Tuple[float, float]] = []
        for match in date_span_pattern.finditer(target_text):
            start = cls._parse_date_point(match.group(1), current_year)
            end = cls._parse_date_point(match.group(2), current_year)
            if start and end and 1980 <= start <= current_year + 1 and start <= end:
                intervals.append((start, end))

        if not intervals:
            return 3.0

        # Merge overlapping time intervals
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if current[0] <= prev_end:
                merged[-1] = (prev_start, max(prev_end, current[1]))
            else:
                merged.append(current)

        total_years = sum(end - start for start, end in merged)
        return min(max(1.0, round(total_years, 1)), 35.0)

    @classmethod
    def _detect_highest_education(cls, text: str) -> str:
        lower_text = text.lower()
        if any(term in lower_text for term in ["ph.d", "phd", "doctorate", "doctor of philosophy"]):
            return "Ph.D. / Doctorate"
        elif any(term in lower_text for term in ["master", "m.s.", "m.sc", "mba", "m.eng", "m.tech"]):
            return "Master's Degree"
        elif any(term in lower_text for term in ["bachelor", "b.s.", "b.sc", "b.a.", "b.eng", "b.tech"]):
            return "Bachelor's Degree"
        elif any(term in lower_text for term in ["associate degree", "associate of"]):
            return "Associate Degree"
        elif "diploma" in lower_text:
            return "Diploma / Certificate"
        return "Bachelor's Degree / Equivalent"

    @classmethod
    def _extract_top_skills(cls, text: str) -> List[str]:
        """Identifies skills from taxonomy and applies canonical synonym normalization."""
        lower_text = f" {text.lower()} "
        found_skills: List[str] = []

        # 1. Check direct skill hits
        for skill in cls.COMMON_SKILLS:
            escaped = re.escape(skill)
            if re.search(rf"(?:\b|\s){escaped}(?:\b|\s|[,\.;])", lower_text):
                found_skills.append(skill.title() if len(skill) > 4 else skill.upper())

        # 2. Check skill synonyms
        for syn, canonical in cls.SKILL_SYNONYMS.items():
            escaped_syn = re.escape(syn)
            if re.search(rf"(?:\b|\s){escaped_syn}(?:\b|\s|[,\.;])", lower_text):
                found_skills.append(canonical)

        # De-duplicate while preserving order
        return list(dict.fromkeys(found_skills))

    @classmethod
    def _extract_summary(cls, sections: List[ResumeSection]) -> Optional[str]:
        for sec in sections:
            if sec.section_type == ResumeSectionType.SUMMARY:
                return sec.content[:600]
        for sec in sections:
            if sec.section_type != ResumeSectionType.HEADER:
                return sec.content[:300]
        return None
