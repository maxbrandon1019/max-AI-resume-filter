"""Test runner script verifying all subsystems using standard library unittest / direct assertion."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_parser import (
    test_parser_plain_text,
    test_sectionizer_entities,
    test_sectionizer_sections,
    test_chunk_creation,
    test_pii_masking,
    test_parser_metadata_and_layout,
    test_word_line_reconstruction,
    test_column_detection_logic,
    test_clean_resume_text_ligatures_and_bullets,
    test_fitz_layout_extraction,
    test_same_line_heading_splitting,
    test_calculate_years_experience_months_overlap
)
from tests.test_retrieval import (
    test_embedding_generation,
    test_hybrid_search_and_ranking,
    test_metadata_filtering,
    test_reranker
)
from tests.test_evaluator import (
    test_jd_analyzer,
    test_candidate_fit_scoring_strong_vs_weak
)
from tests.test_chat import (
    test_chat_agent_citations,
    test_chat_agent_scoped_candidate
)
from tests.test_report_generator import (
    test_html_report_generation,
    test_markdown_dossier_generation,
    test_leaderboard_csv_generation
)


def run_all_tests():
    test_cases = [
        ("Parser: Plain text extraction", test_parser_plain_text),
        ("Parser: Metadata and layout detection", test_parser_metadata_and_layout),
        ("Parser: Word line reconstruction", test_word_line_reconstruction),
        ("Parser: Multi-column spatial layout detection", test_column_detection_logic),
        ("Parser: Unicode ligature, bullet, hyphen & heading normalization", test_clean_resume_text_ligatures_and_bullets),
        ("Parser: PyMuPDF bounding-box layout & column sequencing", test_fitz_layout_extraction),
        ("Sectionizer: Same-line heading splitting (e.g. SKILLS: Python...)", test_same_line_heading_splitting),
        ("Sectionizer: Experience date parser with month-level overlap merging", test_calculate_years_experience_months_overlap),
        ("Sectionizer: Entity extraction (Name, Email, Exp, Edu, Skills)", test_sectionizer_entities),
        ("Sectionizer: Section detection (Summary, Experience, Skills, Edu)", test_sectionizer_sections),
        ("Sectionizer: Chunk document generation", test_chunk_creation),
        ("PII Masker: Candidate anonymization", test_pii_masking),
        ("Embeddings: Generation & normalization", test_embedding_generation),
        ("Retrieval: Hybrid dense + sparse BM25 search & ranking", test_hybrid_search_and_ranking),
        ("Retrieval: Metadata filtering", test_metadata_filtering),
        ("Retrieval: Reranker scoring", test_reranker),
        ("Evaluator: JD requirement decomposition", test_jd_analyzer),
        ("Evaluator: Candidate fit scoring & comparison (Strong vs Weak)", test_candidate_fit_scoring_strong_vs_weak),
        ("Report: Executive HTML brief generation (@media print)", test_html_report_generation),
        ("Report: Markdown dossier generation", test_markdown_dossier_generation),
        ("Report: Candidate leaderboard CSV export", test_leaderboard_csv_generation),
        ("Reader Agent: Conversational QA & Citation generation", test_chat_agent_citations),
        ("Reader Agent: Candidate-scoped queries", test_chat_agent_scoped_candidate)
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("RUNNING MAX'S AI-POWERED RAG RESUME FILTER VERIFICATION SUITE")
    print("=" * 70)

    for name, test_func in test_cases:
        try:
            test_func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (Total: {len(test_cases)})")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL SUBSYSTEM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
