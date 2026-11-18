"""Unit tests for Resume Parser, Sectionizer, and PII Masker."""
from pathlib import Path
from src.ingestion.parser import ResumeParser
from src.ingestion.sectionizer import ResumeSectionizer
from src.ingestion.pii_masker import PIIMasker
from src.models.schemas import ResumeSectionType

SAMPLE_RESUME_TEXT = """
Jane Doe
Seattle, WA | jane.doe@example.com | (555) 123-4567 | linkedin.com/in/janedoe

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 6+ years of experience designing and deploying deep learning pipelines.

TECHNICAL SKILLS
- Languages & Frameworks: Python, PyTorch, TensorFlow, SQL, Docker, FastAPI
- Cloud: AWS (SageMaker, S3), GCP

WORK EXPERIENCE
Senior ML Engineer | AI Tech Labs, Seattle, WA
2020 – Present
- Built and scaled RAG search platform using ChromaDB, FastAPI, and PyTorch.
- Reduced inference latency by 40% with ONNX runtime.

Machine Learning Engineer | Data Innovations, Seattle, WA
2018 – 2020
- Developed NLP classification models using HuggingFace transformers.

EDUCATION
Master of Science in Computer Science | University of Washington (2016 – 2018)
"""


def test_parser_plain_text():
    parsed = ResumeParser.parse_file(SAMPLE_RESUME_TEXT.encode("utf-8"), filename="test_resume.txt")
    assert "Jane Doe" in parsed
    assert "PyTorch" in parsed


def test_sectionizer_entities():
    profile = ResumeSectionizer.sectionize(SAMPLE_RESUME_TEXT, filename="jane_doe.txt")
    
    assert profile.name == "Jane Doe"
    assert profile.email == "jane.doe@example.com"
    assert profile.phone == "(555) 123-4567"
    assert profile.total_years_experience >= 4.0
    assert "Master" in profile.highest_education
    assert any("python" in s.lower() for s in profile.top_skills)
    assert any("pytorch" in s.lower() for s in profile.top_skills)


def test_sectionizer_sections():
    profile = ResumeSectionizer.sectionize(SAMPLE_RESUME_TEXT, filename="jane_doe.txt")
    section_types = [s.section_type for s in profile.sections]
    
    assert ResumeSectionType.SUMMARY in section_types
    assert ResumeSectionType.SKILLS in section_types
    assert ResumeSectionType.EXPERIENCE in section_types
    assert ResumeSectionType.EDUCATION in section_types


def test_chunk_creation():
    profile = ResumeSectionizer.sectionize(SAMPLE_RESUME_TEXT, filename="jane_doe.txt")
    chunks = ResumeSectionizer.create_chunks(profile)
    
    assert len(chunks) >= 3
    assert all(c.candidate_id == profile.candidate_id for c in chunks)
    assert any("overview" in c.chunk_id for c in chunks)


def test_pii_masking():
    profile = ResumeSectionizer.sectionize(SAMPLE_RESUME_TEXT, filename="jane_doe.txt")
    masked = PIIMasker.mask_profile(profile)
    
    assert masked.is_anonymized is True
    assert "Jane Doe" not in masked.name
    assert "Candidate-" in masked.name
    assert masked.email == "[REDACTED_EMAIL]"
    assert masked.phone == "[REDACTED_PHONE]"
    assert "Jane Doe" not in masked.raw_text


def test_parser_metadata_and_layout():
    text, meta = ResumeParser.parse_file_with_metadata(SAMPLE_RESUME_TEXT.encode("utf-8"), filename="test_resume.txt")
    assert "Jane Doe" in text
    assert meta["extension"] == ".txt"
    assert "layout" in meta
    assert meta["layout"] == "single-column"


def test_word_line_reconstruction():
    words = [
        {"text": "Jane", "top": 10.0, "x0": 50.0, "x1": 80.0},
        {"text": "Doe", "top": 10.2, "x0": 85.0, "x1": 110.0},
        {"text": "Software", "top": 30.0, "x0": 50.0, "x1": 100.0},
        {"text": "Engineer", "top": 30.1, "x0": 105.0, "x1": 150.0}
    ]
    reconstructed = ResumeParser._reconstruct_text_from_words(words)
    lines = reconstructed.splitlines()
    assert len(lines) == 2
    assert lines[0] == "Jane Doe"
    assert lines[1] == "Software Engineer"


def test_column_detection_logic():
    class DummyPage:
        width = 600.0
        height = 800.0

        def extract_words(self, x_tolerance=3, y_tolerance=3):
            # Left column (x: 50 to 200)
            left = [
                {"text": "SKILLS", "top": 100.0, "x0": 50.0, "x1": 100.0},
                {"text": "Python", "top": 120.0, "x0": 50.0, "x1": 90.0},
                {"text": "Docker", "top": 140.0, "x0": 50.0, "x1": 90.0},
                {"text": "Kubernetes", "top": 160.0, "x0": 50.0, "x1": 120.0},
                {"text": "FastAPI", "top": 180.0, "x0": 50.0, "x1": 100.0}
            ]
            # Right column (x: 300 to 550)
            right = [
                {"text": "EXPERIENCE", "top": 100.0, "x0": 300.0, "x1": 380.0},
                {"text": "Lead", "top": 120.0, "x0": 300.0, "x1": 330.0},
                {"text": "Architected", "top": 140.0, "x0": 300.0, "x1": 370.0},
                {"text": "Microservices", "top": 160.0, "x0": 300.0, "x1": 380.0},
                {"text": "Distributed", "top": 180.0, "x0": 300.0, "x1": 370.0}
            ]
            return left + right

        def extract_text(self):
            return "dummy text"

    page = DummyPage()
    text, is_multi = ResumeParser._extract_page_with_column_detection(page)
    assert is_multi is True
    # Column 1 should precede Column 2 in text output (not horizontally mixed!)
    assert text.index("FastAPI") < text.index("EXPERIENCE")


def test_clean_resume_text_ligatures_and_bullets():
    raw_dirty = "Speci\ufb01cation and \ufb02exibility \uf0b7 Micro-\nservices architec-\nture.  S K I L L S  "
    cleaned = ResumeParser.clean_resume_text(raw_dirty)

    # Ligatures resolved
    assert "Specification" in cleaned
    assert "flexibility" in cleaned
    # Hyphenation across newlines stitched
    assert "Microservices" in cleaned
    assert "architecture" in cleaned
    # Bullet normalized
    assert "•" in cleaned
    # Spaced heading normalized
    assert "SKILLS" in cleaned


def test_fitz_layout_extraction():
    try:
        import fitz
    except ImportError:
        return

    doc = fitz.open()
    # 600 x 800 page
    page = doc.new_page(width=600, height=800)

    # Insert top header
    page.insert_text((50, 40), "ALEX MERCER | LEAD ARCHITECT", fontsize=12)

    # Insert left column (x: 50, y: 120-250)
    page.insert_text((50, 120), "CORE SKILLS", fontsize=11)
    page.insert_text((50, 150), "Python", fontsize=10)
    page.insert_text((50, 170), "Kubernetes", fontsize=10)
    page.insert_text((50, 190), "Docker", fontsize=10)

    # Insert right column (x: 320, y: 120-250)
    page.insert_text((320, 120), "PROFESSIONAL EXPERIENCE", fontsize=11)
    page.insert_text((320, 150), "Staff Architect at CloudOps", fontsize=10)
    page.insert_text((320, 170), "Designed real-time event pipelines", fontsize=10)

    text, is_multi = ResumeParser._extract_fitz_page_lines(page)
    doc.close()

    assert is_multi is True
    assert "ALEX MERCER" in text
    assert "CORE SKILLS" in text
    assert "PROFESSIONAL EXPERIENCE" in text
    # Column 1 should precede Column 2 in sequential output
    assert text.index("Docker") < text.index("PROFESSIONAL EXPERIENCE")


def test_same_line_heading_splitting():
    raw_lines = [
        "Jane Doe",
        "SKILLS: Python, Go, Docker, Kubernetes, FastAPI",
        "EXPERIENCE: Senior Engineer at Google"
    ]
    processed = ResumeSectionizer._preprocess_lines(raw_lines)
    assert "SKILLS" in processed
    assert "Python, Go, Docker, Kubernetes, FastAPI" in processed
    assert "EXPERIENCE" in processed
    assert "Senior Engineer at Google" in processed

    # Verify sectionize catches it
    full_text = "\n".join(raw_lines)
    profile = ResumeSectionizer.sectionize(full_text, filename="jane_split.txt")
    sec_types = [s.section_type for s in profile.sections]
    assert ResumeSectionType.SKILLS in sec_types
    assert ResumeSectionType.EXPERIENCE in sec_types


def test_calculate_years_experience_months_overlap():
    text = """
    EXPERIENCE
    Senior Software Engineer | Tech Corp
    Jan 2020 – Dec 2022
    - Developed microservices

    Staff Engineer | Cloud Labs
    Jun 2021 – Jun 2023
    - Led platform team
    """
    profile = ResumeSectionizer.sectionize(text, filename="experience_overlap.txt")
    # Jan 2020 to Jun 2023 is 3.5 years (3 years 5-6 months)
    # Without overlapping merge, naive summing would yield 3 + 2 = 5 years!
    assert 3.2 <= profile.total_years_experience <= 3.7

