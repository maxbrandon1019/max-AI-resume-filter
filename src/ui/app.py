"""Streamlit Dashboard for Max's AI-Powered RAG Resume Filter."""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from src.models.schemas import CandidateProfile, JDRequirements, FitEvaluation, ResumeSectionType, WorkExperience
from src.ingestion.parser import ResumeParser
from src.ingestion.sectionizer import ResumeSectionizer
from src.ingestion.pii_masker import PIIMasker
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import HybridVectorStore
from src.evaluator.llm_client import LLMClient
from src.evaluator.jd_analyzer import JDAnalyzer
from src.evaluator.scorer import CandidateScorer
from src.evaluator.report_generator import ReportGenerator
from src.reader.chat_agent import ResumeChatAgent
from src.ui.charts import generate_radar_chart_svg, generate_comparison_radar_chart_svg

# Page configuration
st.set_page_config(
    page_title="Max's AI-Powered RAG Resume Filter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, theme-adaptive minimalist styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(127, 127, 127, 0.06);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }
    .score-badge-high {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge-med {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge-low {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .skill-pill {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.12);
        color: #6366F1;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 2px;
        font-weight: 500;
    }
    .missing-pill {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 2px;
        text-decoration: line-through;
    }
    .diff-pill-a {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 2px;
        font-weight: 600;
    }
    .diff-pill-b {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 2px;
        font-weight: 600;
    }
    .diff-pill-both {
        display: inline-block;
        background-color: rgba(127, 127, 127, 0.12);
        color: currentColor;
        border: 1px solid rgba(127, 127, 127, 0.25);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 2px;
        opacity: 0.85;
    }
    .comparison-box {
        background-color: rgba(127, 127, 127, 0.05);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "candidates" not in st.session_state:
        st.session_state.candidates: Dict[str, CandidateProfile] = {}
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = HybridVectorStore()
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient()
    if "evaluations" not in st.session_state:
        st.session_state.evaluations: Dict[str, FitEvaluation] = {}
    if "current_jd" not in st.session_state:
        st.session_state.current_jd: Optional[JDRequirements] = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []
    if "pii_masking" not in st.session_state:
        st.session_state.pii_masking = False


init_session_state()


def reindex_all_candidates():
    """Rebuilds the vector store index from current candidate profiles."""
    st.session_state.vector_store.clear()
    for cand in st.session_state.candidates.values():
        profile = PIIMasker.mask_profile(cand) if st.session_state.pii_masking else cand
        chunks = ResumeSectionizer.create_chunks(profile)
        st.session_state.vector_store.add_chunks(chunks)


def load_sample_dataset():
    """Loads bundled sample resumes and sample job descriptions."""
    sample_resumes_dir = PROJECT_ROOT / "data" / "sample_resumes"
    if sample_resumes_dir.exists():
        for file_path in sample_resumes_dir.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
            profile = ResumeSectionizer.sectionize(text, filename=file_path.name)
            st.session_state.candidates[profile.candidate_id] = profile

    reindex_all_candidates()

    # Load default JD
    sample_jd_path = PROJECT_ROOT / "data" / "sample_jds" / "senior_ai_python_engineer.txt"
    if sample_jd_path.exists():
        jd_text = sample_jd_path.read_text(encoding="utf-8")
        analyzer = JDAnalyzer(st.session_state.llm_client)
        st.session_state.current_jd = analyzer.analyze(jd_text)


# ==============================================================================
# SIDEBAR: Configuration & Talent Pool Status
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Control Center")
    st.caption("RAG Resume Intelligence")

    st.subheader("🤖 AI Provider Settings")
    provider_choice = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "mock (offline)"],
        index=0 if os.getenv("GEMINI_API_KEY") else (1 if os.getenv("OPENAI_API_KEY") else 2)
    )

    api_key_input = st.text_input(
        f"{provider_choice.capitalize()} API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", "") if provider_choice == "gemini" else os.getenv("OPENAI_API_KEY", ""),
        help="Enter API key or set it in your .env file."
    )

    if api_key_input:
        if provider_choice == "gemini":
            os.environ["GEMINI_API_KEY"] = api_key_input
            os.environ["LLM_PROVIDER"] = "gemini"
        elif provider_choice == "openai":
            os.environ["OPENAI_API_KEY"] = api_key_input
            os.environ["LLM_PROVIDER"] = "openai"
        st.session_state.llm_client = LLMClient(provider=provider_choice, api_key=api_key_input)
        st.session_state.vector_store.embedding_manager = EmbeddingManager(provider=provider_choice, api_key=api_key_input)

    # Status Indicator
    if st.session_state.llm_client.is_live_llm_available:
        st.success(f"🟢 Active: {provider_choice.upper()} connected")
    else:
        st.info("🟡 Local Hybrid Engine (Offline / Zero-API mode)")

    st.divider()

    st.subheader("🛡️ Bias-Free Screening")
    pii_toggle = st.toggle(
        "Enable PII Anonymization",
        value=st.session_state.pii_masking,
        help="Mask candidate names, emails, phone numbers, and links during screening."
    )
    if pii_toggle != st.session_state.pii_masking:
        st.session_state.pii_masking = pii_toggle
        reindex_all_candidates()
        st.rerun()

    st.divider()

    st.subheader("📊 Talent Index Stats")
    c1, c2 = st.columns(2)
    c1.metric("Resumes", len(st.session_state.candidates))
    c2.metric("RAG Chunks", len(st.session_state.vector_store.chunks))

    col_load, col_clear = st.columns(2)
    if col_load.button("📥 Load Samples", use_container_width=True):
        load_sample_dataset()
        st.toast("Sample resumes & JDs loaded successfully!", icon="✅")
        st.rerun()

    if col_clear.button("🗑️ Reset Pool", use_container_width=True):
        st.session_state.candidates.clear()
        st.session_state.vector_store.clear()
        st.session_state.evaluations.clear()
        st.session_state.chat_history.clear()
        st.toast("Talent pool reset.", icon="🧹")
        st.rerun()


# ==============================================================================
# MAIN PAGE HEADER & TABS
# ==============================================================================
st.markdown('<div class="main-title">📄 Max\'s AI-Powered RAG Resume Filter</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Section-aware parsing, explainable candidate-JD fit scoring, and conversational talent intelligence</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "📁 1. Resume Pool",
    "🎯 2. JD & Match Leaderboard",
    "🔍 3. Deep-Dive Gap Analysis",
    "💬 4. Interactive RAG Chat"
])


# ==============================================================================
# TAB 1: RESUME INGESTION & POOL
# ==============================================================================
with tabs[0]:
    st.subheader("Upload Candidate Resumes")
    col_up, col_eng = st.columns([2, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload resumes in PDF, DOCX, or TXT format (supports multiple files)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )
    with col_eng:
        selected_engine = st.selectbox(
            "PDF Extraction Engine",
            options=["auto", "fitz", "gemini", "pdfplumber"],
            index=0,
            help=(
                "• auto: PyMuPDF bounding-box column sorting + Gemini OCR for scans\n"
                "• fitz: PyMuPDF layout-aware engine (fast, prevents column interleaving)\n"
                "• gemini: Multimodal Vision OCR (scanned / visual PDFs)\n"
                "• pdfplumber: Traditional coordinate-based layout engine"
            )
        )

    if uploaded_files:
        with st.spinner(f"Parsing and sectionizing {len(uploaded_files)} resumes..."):
            new_added = 0
            for file in uploaded_files:
                file_bytes = file.read()
                raw_text, meta = ResumeParser.parse_file_with_metadata(
                    file_bytes, filename=file.name, engine=selected_engine
                )
                layout_type = meta.get("layout", "single-column")
                engine_used = meta.get("engine", "standard")
                is_scanned = meta.get("is_scanned", False)
                if raw_text.strip():
                    profile = ResumeSectionizer.sectionize(
                        raw_text,
                        filename=file.name,
                        layout_type=layout_type,
                        extraction_engine=engine_used,
                        is_scanned=is_scanned
                    )
                    st.session_state.candidates[profile.candidate_id] = profile
                    new_added += 1

            if new_added > 0:
                reindex_all_candidates()
                st.toast(f"Successfully processed {new_added} resumes!", icon="🎉")

    st.divider()

    st.subheader(f"Current Talent Pool ({len(st.session_state.candidates)} Candidates)")

    if not st.session_state.candidates:
        st.info("No resumes uploaded yet. Upload files above or click **'📥 Load Samples'** in the sidebar to get started instantly.")
    else:
        # Quick Overview Table across all candidates
        st.markdown("##### 📋 Talent Pool Quick Summary Table")
        pool_rows = []
        for cand in st.session_state.candidates.values():
            disp = PIIMasker.mask_profile(cand) if st.session_state.pii_masking else cand
            latest_role = cand.work_experiences[0].title if cand.work_experiences else "—"
            if cand.work_experiences and cand.work_experiences[0].company:
                latest_role += f" @ {cand.work_experiences[0].company}"
            pool_rows.append({
                "Name": disp.name,
                "Email": disp.email or "—",
                "Phone": disp.phone or "—",
                "Location": disp.location or "—",
                "Experience": f"{disp.total_years_experience or 0:.1f} yrs",
                "Education": disp.highest_education or "—",
                "Core Skills": ", ".join(disp.top_skills[:5]) + ("..." if len(disp.top_skills) > 5 else ""),
                "Latest Position": latest_role
            })
        st.dataframe(pd.DataFrame(pool_rows), use_container_width=True, hide_index=True)

        st.markdown("##### 🗂️ Detailed Candidate Profiles")
        for cid, cand in list(st.session_state.candidates.items()):
            display_profile = PIIMasker.mask_profile(cand) if st.session_state.pii_masking else cand
            layout_icon = "📰 Multi-Column / Sidebar" if getattr(cand, "layout_type", "single-column") == "multi-column" else "📑 Standard 1-Column"
            engine_label = {
                "fitz": "⚡ PyMuPDF Layout",
                "gemini": "🤖 Gemini Vision OCR",
                "pdfplumber": "📐 pdfplumber Spatial",
                "pypdf": "📄 pypdf Stream",
                "standard": "⚙️ Standard"
            }.get(getattr(cand, "extraction_engine", "standard"), getattr(cand, "extraction_engine", "standard"))

            word_count = len(cand.raw_text.split()) if cand.raw_text else 0

            with st.expander(f"👤 {display_profile.name} — {display_profile.highest_education} (~{int(display_profile.total_years_experience or 0)} yrs exp) [{cand.filename}]"):
                if getattr(cand, "is_scanned", False):
                    st.warning("⚠️ **Scanned / Visual Document Detected:** Native text stream was minimal (<35 words). Content was recovered using Vision OCR / fallbacks.")

                # 1. SEPARATED CONTACT & LOCATION HEADER
                st.markdown("#### 👤 Candidate Contact & Details")
                c_mail, c_phone, c_loc, c_exp, c_edu = st.columns([1.3, 1.1, 1.2, 0.9, 1.3])
                with c_mail:
                    st.markdown(f"📧 **Email:**<br>`{display_profile.email or 'Not specified'}`", unsafe_allow_html=True)
                with c_phone:
                    st.markdown(f"📱 **Phone:**<br>`{display_profile.phone or 'Not specified'}`", unsafe_allow_html=True)
                with c_loc:
                    st.markdown(f"📍 **Location:**<br>`{display_profile.location or 'Not specified'}`", unsafe_allow_html=True)
                with c_exp:
                    st.markdown(f"⏳ **Experience:**<br>`{display_profile.total_years_experience} years`", unsafe_allow_html=True)
                with c_edu:
                    st.markdown(f"🎓 **Education:**<br>`{display_profile.highest_education}`", unsafe_allow_html=True)

                if display_profile.links and not st.session_state.pii_masking:
                    links_md = " • ".join([f"[{url}]({url if url.startswith('http') else 'https://' + url})" for url in display_profile.links[:4]])
                    st.markdown(f"🔗 **Web Links:** {links_md}")

                st.divider()

                # 2. SEPARATED CORE SKILLS
                st.markdown(f"#### 🛠️ Detected Skills ({len(display_profile.top_skills)})")
                if display_profile.top_skills:
                    skill_badges = "".join([f'<span class="skill-pill">{s}</span>' for s in display_profile.top_skills])
                    st.markdown(skill_badges, unsafe_allow_html=True)
                else:
                    st.caption("No technical skills explicitly identified.")

                st.divider()

                # 3. SEPARATED STRUCTURED WORK EXPERIENCES
                st.markdown(f"#### 💼 Work Experience ({len(display_profile.work_experiences)} Roles Extracted)")
                if display_profile.work_experiences:
                    for idx, exp in enumerate(display_profile.work_experiences, 1):
                        dur_str = f" • ~{exp.duration_years} yrs" if exp.duration_years else ""
                        comp_str = f" @ **{exp.company}**" if exp.company else ""
                        loc_str = f" ({exp.location})" if exp.location else ""
                        
                        st.markdown(f"**{idx}. {exp.title}**{comp_str}{loc_str}")
                        st.caption(f"🗓️ {exp.date_range}{dur_str}")
                        if exp.highlights:
                            for h in exp.highlights:
                                st.markdown(f"- {h}")
                        elif exp.description:
                            st.markdown(exp.description)
                        st.write("")
                else:
                    # Fallback to experience section text
                    exp_sec = next((s for s in display_profile.sections if s.section_type == ResumeSectionType.EXPERIENCE), None)
                    if exp_sec:
                        st.markdown(exp_sec.content)
                    else:
                        st.caption("No work experience entries parsed.")

                # 4. SUMMARY & RAW RESUME SECTIONS
                with st.expander("📄 View Full Summary & Raw Resume Text", expanded=False):
                    if display_profile.summary:
                        st.markdown(f"**Executive Summary:**\n> {display_profile.summary}\n")
                    st.text(display_profile.raw_text)


# ==============================================================================
# TAB 2: JOB DESCRIPTION & MATCH LEADERBOARD
# ==============================================================================
with tabs[1]:
    st.subheader("1. Configure Job Description (JD)")
    
    sample_jds_dir = PROJECT_ROOT / "data" / "sample_jds"
    jd_options = {}
    if sample_jds_dir.exists():
        for jd_file in sample_jds_dir.glob("*.txt"):
            jd_options[jd_file.stem.replace("_", " ").title()] = jd_file.read_text(encoding="utf-8")

    jd_selection = st.selectbox(
        "Select a Pre-built Job Template or Paste Custom JD",
        options=["Custom Job Description"] + list(jd_options.keys())
    )

    initial_text = ""
    if jd_selection != "Custom Job Description":
        initial_text = jd_options[jd_selection]
    elif st.session_state.current_jd:
        initial_text = st.session_state.current_jd.raw_jd_text

    jd_text_input = st.text_area(
        "Job Description Text",
        value=initial_text,
        height=200,
        placeholder="Paste full Job Description here..."
    )

    c_btn1, c_btn2 = st.columns([1, 4])
    if c_btn1.button("⚡ Analyze & Match", type="primary", use_container_width=True):
        if not jd_text_input.strip():
            st.error("Please enter a Job Description.")
        elif not st.session_state.candidates:
            st.error("Please upload or load candidate resumes in Tab 1 first.")
        else:
            with st.spinner("Deconstructing JD requirements and running hybrid RAG matching..."):
                analyzer = JDAnalyzer(st.session_state.llm_client)
                st.session_state.current_jd = analyzer.analyze(jd_text_input)

                scorer = CandidateScorer(
                    llm_client=st.session_state.llm_client,
                    vector_store=st.session_state.vector_store
                )
                evaluations = {}
                for cand in st.session_state.candidates.values():
                    profile = PIIMasker.mask_profile(cand) if st.session_state.pii_masking else cand
                    eval_result = scorer.evaluate(profile, st.session_state.current_jd)
                    evaluations[cand.candidate_id] = eval_result

                st.session_state.evaluations = evaluations
                st.toast(f"Evaluated {len(evaluations)} candidates successfully!", icon="🎯")

    if st.session_state.current_jd:
        jd = st.session_state.current_jd
        with st.expander(f"📋 Extracted Target Criteria: {jd.job_title}", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            col_a.write(f"**Min Experience:** {jd.min_years_experience} years")
            col_b.write(f"**Education:** {jd.preferred_education}")
            col_c.write(f"**Required Skills Count:** {len(jd.required_skills)}")
            st.write(f"**Required Skills:** {', '.join(jd.required_skills)}")
            st.write(f"**Preferred Skills:** {', '.join(jd.preferred_skills)}")

    st.divider()

    # LEADERBOARD
    st.subheader("2. Candidate Match Leaderboard")

    if not st.session_state.evaluations:
        st.info("Run candidate matching above to view the ranked leaderboard.")
    else:
        # Sort candidates descending by overall score initially
        sorted_evals = sorted(st.session_state.evaluations.values(), key=lambda x: x.overall_score, reverse=True)

        # Filter and Sorting Controls
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.8, 2.2, 1.5])
        with f_col1:
            min_score_filter = st.slider("🎯 Min Overall Score", 0, 100, 0, step=5)
        with f_col2:
            tier_filter = st.multiselect(
                "🏷️ Fit Tiers",
                options=["Strong Fit", "Moderate Fit", "Weak Fit"],
                default=["Strong Fit", "Moderate Fit", "Weak Fit"]
            )
        with f_col3:
            search_filter = st.text_input("🔍 Search Candidate / Skill", placeholder="Filter by name, skill, degree...")
        with f_col4:
            sort_by_filter = st.selectbox(
                "Sort Leaderboard By",
                options=["Overall Score", "Technical Score", "Experience Score", "Skills Matched"]
            )

        # Apply filtering
        filtered_evals = []
        for ev in sorted_evals:
            if ev.overall_score < min_score_filter:
                continue
            if tier_filter and ev.fit_tier not in tier_filter:
                continue
            if search_filter.strip():
                q = search_filter.strip().lower()
                cand = st.session_state.candidates.get(ev.candidate_id)
                skills_str = " ".join(s.skill_name for s in ev.skills_matrix)
                corpus = f"{ev.candidate_name} {ev.fit_summary} {skills_str} {cand.highest_education if cand else ''}".lower()
                if q not in corpus:
                    continue
            filtered_evals.append(ev)

        # Apply sorting
        if sort_by_filter == "Technical Score":
            filtered_evals.sort(key=lambda x: x.technical_score, reverse=True)
        elif sort_by_filter == "Experience Score":
            filtered_evals.sort(key=lambda x: x.experience_score, reverse=True)
        elif sort_by_filter == "Skills Matched":
            filtered_evals.sort(key=lambda x: x.matched_skills_count, reverse=True)
        else:
            filtered_evals.sort(key=lambda x: x.overall_score, reverse=True)

        st.caption(f"Showing **{len(filtered_evals)}** of **{len(sorted_evals)}** evaluated candidates")

        if not filtered_evals:
            st.info("No candidates match your current filter criteria. Adjust the sliders or search text above.")
        else:
            table_rows = []
            for rank, ev in enumerate(filtered_evals, 1):
                cand = st.session_state.candidates.get(ev.candidate_id)
                display_name = ev.candidate_name
                
                table_rows.append({
                    "Rank": f"#{rank}",
                    "Candidate": display_name,
                    "Fit Tier": ev.fit_tier,
                    "Overall Score": f"{ev.overall_score} / 100",
                    "Technical Score": f"{ev.technical_score}%",
                    "Experience Score": f"{ev.experience_score}%",
                    "Skills Matched": f"{ev.matched_skills_count} / {ev.total_required_skills_count}",
                    "Filename": cand.filename if cand else "N/A"
                })

            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Leaderboard CSV Export
            csv_data = ReportGenerator.generate_leaderboard_csv(
                filtered_evals,
                st.session_state.candidates
            )
            st.download_button(
                label=f"📥 Export Filtered Leaderboard ({len(filtered_evals)} Candidates CSV)",
                data=csv_data,
                file_name=f"candidate_leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download the filtered candidate list with fit scores and gap summaries for ATS export."
            )

            # Visual Candidate Cards
            st.markdown("### 🏆 Top Candidate Snapshots")
            cols = st.columns(min(3, len(filtered_evals)))
            for i, ev in enumerate(filtered_evals[:3]):
                with cols[i]:
                    badge_class = "score-badge-high" if ev.overall_score >= 75 else ("score-badge-med" if ev.overall_score >= 55 else "score-badge-low")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>#{i+1} {ev.candidate_name}</h4>
                        <span class="{badge_class}">{ev.fit_tier} • {ev.overall_score}/100</span>
                        <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.85;">
                            <b>Tech:</b> {ev.technical_score}% | <b>Exp:</b> {ev.experience_score}%
                        </p>
                        <p style="font-size: 0.85rem; text-align: left; opacity: 0.8;">{ev.fit_summary[:140]}...</p>
                    </div>
                    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: DEEP-DIVE CANDIDATE & GAP ANALYSIS
# ==============================================================================
with tabs[2]:
    st.subheader("Candidate Fit & Gap Analysis Deep-Dive")

    if not st.session_state.evaluations:
        st.info("Run candidate matching in Tab 2 to inspect detailed candidate breakdowns.")
    else:
        cand_choice_map = {ev.candidate_name: ev.candidate_id for ev in st.session_state.evaluations.values()}
        cand_names = list(cand_choice_map.keys())

        view_mode = st.radio(
            "Analysis View",
            options=["🔍 Single Candidate Deep-Dive", "⚔️ Head-to-Head Comparison (A vs B)"],
            horizontal=True,
            help="Switch between inspecting an individual candidate and comparing two finalists side-by-side."
        )

        st.divider()

        if view_mode == "🔍 Single Candidate Deep-Dive":
            selected_name = st.selectbox("Select Candidate to Inspect", options=cand_names)
            selected_cid = cand_choice_map[selected_name]
            ev = st.session_state.evaluations[selected_cid]
            cand = st.session_state.candidates[selected_cid]
            display_cand = PIIMasker.mask_profile(cand) if st.session_state.pii_masking else cand

            # Separated Candidate Contact & Location Strip
            c_info1, c_info2, c_info3 = st.columns(3)
            c_info1.markdown(f"📧 **Email:** `{display_cand.email or 'Not specified'}`")
            c_info2.markdown(f"📱 **Phone:** `{display_cand.phone or 'Not specified'}`")
            c_info3.markdown(f"📍 **Location:** `{display_cand.location or 'Not specified'}`")

            st.write("")

            col_left_card, col_right_radar = st.columns([1.1, 1])

            with col_left_card:
                # Top Score Metrics
                m1, m2 = st.columns(2)
                m1.metric("Overall Fit", f"{ev.overall_score}/100", delta=ev.fit_tier)
                m2.metric("Technical Match", f"{ev.technical_score}%")
                m3, m4 = st.columns(2)
                m3.metric("Experience Match", f"{ev.experience_score}%")
                m4.metric("Education Fit", f"{ev.education_score}%")

                st.info(f"**Executive Summary:** {ev.fit_summary}")

            with col_right_radar:
                skill_cov = min(100.0, (ev.matched_skills_count / max(1, ev.total_required_skills_count)) * 100.0)
                radar_scores = {
                    "Overall Fit": float(ev.overall_score),
                    "Technical": float(ev.technical_score),
                    "Experience": float(ev.experience_score),
                    "Education": float(ev.education_score),
                    "Skills Cov.": float(skill_cov)
                }
                radar_svg = generate_radar_chart_svg(display_cand.name, radar_scores)
                st.markdown(radar_svg, unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("### 🧩 Skill Match Matrix")
                matched_skills = [sm for sm in ev.skills_matrix if sm.matched]
                missing_skills = [sm for sm in ev.skills_matrix if not sm.matched]

                st.write(f"**✅ Matched Skills ({len(matched_skills)}):**")
                p_matched = "".join([f'<span class="skill-pill">✓ {sm.skill_name}</span>' for sm in matched_skills])
                st.markdown(p_matched if p_matched else "*None matched*", unsafe_allow_html=True)

                st.write(f"**❌ Missing / Unverified Skills ({len(missing_skills)}):**")
                p_missing = "".join([f'<span class="missing-pill">✕ {sm.skill_name}</span>' for sm in missing_skills])
                st.markdown(p_missing if p_missing else "*None missing*", unsafe_allow_html=True)

                st.markdown("### 🌟 Key Strengths")
                for strength in ev.strengths:
                    st.markdown(f"- 🟢 {strength}")

            with col_right:
                st.markdown("### ⚠️ Gaps & Areas of Concern")
                for gap in ev.gaps:
                    st.markdown(f"- 🟠 {gap}")

                st.markdown("### 🎯 Recommended Interview Questions")
                st.caption("Tailored prompts targeting this candidate's specific profile and skill gaps:")
                for idx, q in enumerate(ev.tailored_interview_questions, 1):
                    st.markdown(f"**{idx}.** {q}")

            # Structured Work Experience section in Tab 3
            if display_cand.work_experiences:
                st.markdown(f"### 💼 Structured Career Experience ({len(display_cand.work_experiences)} Roles)")
                for idx, exp in enumerate(display_cand.work_experiences, 1):
                    dur_str = f" • ~{exp.duration_years} yrs" if exp.duration_years else ""
                    comp_str = f" @ **{exp.company}**" if exp.company else ""
                    loc_str = f" ({exp.location})" if exp.location else ""
                    with st.expander(f"{idx}. {exp.title}{comp_str} ({exp.date_range}{dur_str})", expanded=(idx == 1)):
                        if exp.highlights:
                            for h in exp.highlights:
                                st.markdown(f"- {h}")
                        elif exp.description:
                            st.markdown(exp.description)

            st.divider()
            st.markdown("### 📤 Executive Hiring Packets & Reports")
            col_exp1, col_exp2 = st.columns(2)

            html_brief = ReportGenerator.generate_executive_html(
                display_cand,
                ev,
                st.session_state.current_jd
            )
            col_exp1.download_button(
                label="📄 Download Executive Brief (Printable HTML / PDF)",
                data=html_brief,
                file_name=f"executive_brief_{ev.candidate_name.replace(' ', '_').lower()}.html",
                mime="text/html",
                use_container_width=True,
                help="Opens in any web browser. Press Ctrl+P (Save as PDF) for a ready-to-print 1-2 page hiring manager brief."
            )

            md_dossier = ReportGenerator.generate_markdown_dossier(
                display_cand,
                ev,
                st.session_state.current_jd
            )
            col_exp2.download_button(
                label="📝 Download Markdown Dossier (.md)",
                data=md_dossier,
                file_name=f"dossier_{ev.candidate_name.replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Clean markdown formatted for Notion, Slack, or internal applicant tracking systems."
            )

            with st.expander("👁️ Live Preview: Executive Printable Brief", expanded=False):
                st.components.v1.html(html_brief, height=600, scrolling=True)

            st.divider()
            st.markdown("### 📑 Full Resume Content")
            with st.expander("View Full Resume Text", expanded=False):
                st.text(display_cand.raw_text)

        else:
            # HEAD-TO-HEAD COMPARISON MODE (A vs B)
            st.markdown("### ⚔️ Candidate Head-to-Head Comparative Matrix")
            st.caption("Compare qualifications, skills coverage, and trade-offs between two top contenders.")

            sel_col_a, sel_col_b = st.columns(2)
            default_b = 1 if len(cand_names) > 1 else 0
            with sel_col_a:
                name_a = st.selectbox("Candidate A (Baseline)", options=cand_names, index=0)
            with sel_col_b:
                name_b = st.selectbox("Candidate B (Challenger)", options=cand_names, index=default_b)

            cid_a = cand_choice_map[name_a]
            cid_b = cand_choice_map[name_b]
            ev_a = st.session_state.evaluations[cid_a]
            ev_b = st.session_state.evaluations[cid_b]
            cand_a = st.session_state.candidates[cid_a]
            cand_b = st.session_state.candidates[cid_b]
            disp_a = PIIMasker.mask_profile(cand_a) if st.session_state.pii_masking else cand_a
            disp_b = PIIMasker.mask_profile(cand_b) if st.session_state.pii_masking else cand_b

            # Scorecard Comparison Header
            sc1, sc2, sc3, sc4 = st.columns(4)
            delta_overall = ev_a.overall_score - ev_b.overall_score
            delta_tech = ev_a.technical_score - ev_b.technical_score
            delta_exp = ev_a.experience_score - ev_b.experience_score
            delta_edu = ev_a.education_score - ev_b.education_score

            sc1.metric(
                "Overall Fit",
                f"{ev_a.overall_score} vs {ev_b.overall_score}",
                delta=f"{delta_overall:+} pts (A)" if delta_overall != 0 else "Tied"
            )
            sc2.metric(
                "Technical Match",
                f"{ev_a.technical_score}% vs {ev_b.technical_score}%",
                delta=f"{delta_tech:+}% (A)" if delta_tech != 0 else "Tied"
            )
            sc3.metric(
                "Experience Match",
                f"{ev_a.experience_score}% vs {ev_b.experience_score}%",
                delta=f"{delta_exp:+}% (A)" if delta_exp != 0 else "Tied"
            )
            sc4.metric(
                "Education Score",
                f"{ev_a.education_score}% vs {ev_b.education_score}%",
                delta=f"{delta_edu:+}% (A)" if delta_edu != 0 else "Tied"
            )

            # Dual Radar Comparison Chart
            st.markdown("#### 📊 Overlapping Fit Radar")
            skill_cov_a = min(100.0, (ev_a.matched_skills_count / max(1, ev_a.total_required_skills_count)) * 100.0)
            skill_cov_b = min(100.0, (ev_b.matched_skills_count / max(1, ev_b.total_required_skills_count)) * 100.0)

            scores_a = {
                "Overall Fit": float(ev_a.overall_score),
                "Technical": float(ev_a.technical_score),
                "Experience": float(ev_a.experience_score),
                "Education": float(ev_a.education_score),
                "Skills Cov.": float(skill_cov_a)
            }
            scores_b = {
                "Overall Fit": float(ev_b.overall_score),
                "Technical": float(ev_b.technical_score),
                "Experience": float(ev_b.experience_score),
                "Education": float(ev_b.education_score),
                "Skills Cov.": float(skill_cov_b)
            }
            comp_chart_svg = generate_comparison_radar_chart_svg(disp_a.name, scores_a, disp_b.name, scores_b)
            st.markdown(comp_chart_svg, unsafe_allow_html=True)

            # Skill Differential Analysis
            st.markdown("#### 🧩 Skill Differential Analysis")
            skills_dict_a = {sm.skill_name.lower(): sm.skill_name for sm in ev_a.skills_matrix if sm.matched}
            skills_dict_b = {sm.skill_name.lower(): sm.skill_name for sm in ev_b.skills_matrix if sm.matched}

            set_a = set(skills_dict_a.keys())
            set_b = set(skills_dict_b.keys())

            shared_keys = set_a & set_b
            a_only_keys = set_a - set_b
            b_only_keys = set_b - set_a

            all_req_skills = {sm.skill_name.lower(): sm.skill_name for sm in ev_a.skills_matrix}
            mutual_missing = set(all_req_skills.keys()) - (set_a | set_b)

            c_diff1, c_diff2 = st.columns(2)
            with c_diff1:
                st.markdown(f"**🔹 Candidate A Exclusive Advantages ({len(a_only_keys)}):**")
                pills_a = "".join([f'<span class="diff-pill-a">★ {skills_dict_a[k]}</span>' for k in sorted(a_only_keys)])
                st.markdown(pills_a if pills_a else "*No exclusive skills*", unsafe_allow_html=True)

                st.markdown(f"**🤝 Shared Matched Skills ({len(shared_keys)}):**")
                pills_shared = "".join([f'<span class="diff-pill-both">✓ {skills_dict_a[k]}</span>' for k in sorted(shared_keys)])
                st.markdown(pills_shared if pills_shared else "*No shared skills*", unsafe_allow_html=True)

            with c_diff2:
                st.markdown(f"**🟢 Candidate B Exclusive Advantages ({len(b_only_keys)}):**")
                pills_b = "".join([f'<span class="diff-pill-b">★ {skills_dict_b[k]}</span>' for k in sorted(b_only_keys)])
                st.markdown(pills_b if pills_b else "*No exclusive skills*", unsafe_allow_html=True)

                st.markdown(f"**❌ Mutual Missing Skills ({len(mutual_missing)}):**")
                pills_miss = "".join([f'<span class="missing-pill">✕ {all_req_skills[k]}</span>' for k in sorted(mutual_missing)])
                st.markdown(pills_miss if pills_miss else "*None missing from both*", unsafe_allow_html=True)

            # Side-by-Side Strengths & Concerns
            st.divider()
            st.markdown("#### ⚖️ Side-by-Side Strengths & Concerns")
            col_a_details, col_b_details = st.columns(2)

            with col_a_details:
                st.markdown(f"##### 👤 Candidate A: {disp_a.name}")
                st.caption(f"**Education:** {disp_a.highest_education} | **Experience:** ~{int(disp_a.total_years_experience or 0)} yrs")
                st.markdown(f"*{ev_a.fit_summary}*")
                st.markdown("**Top Strengths:**")
                for s in ev_a.strengths[:3]:
                    st.markdown(f"- 🟢 {s}")
                st.markdown("**Areas of Concern:**")
                for g in ev_a.gaps[:3]:
                    st.markdown(f"- 🟠 {g}")

            with col_b_details:
                st.markdown(f"##### 👤 Candidate B: {disp_b.name}")
                st.caption(f"**Education:** {disp_b.highest_education} | **Experience:** ~{int(disp_b.total_years_experience or 0)} yrs")
                st.markdown(f"*{ev_b.fit_summary}*")
                st.markdown("**Top Strengths:**")
                for s in ev_b.strengths[:3]:
                    st.markdown(f"- 🟢 {s}")
                st.markdown("**Areas of Concern:**")
                for g in ev_b.gaps[:3]:
                    st.markdown(f"- 🟠 {g}")

            # Comparative Hiring Recommendation Takeaway
            st.divider()
            st.markdown("#### 💡 Comparative Hiring Takeaway")
            winner = disp_a.name if ev_a.overall_score >= ev_b.overall_score else disp_b.name
            if abs(ev_a.overall_score - ev_b.overall_score) <= 3:
                takeaway = (
                    f"**Close Call:** {disp_a.name} ({ev_a.overall_score}/100) and {disp_b.name} ({ev_b.overall_score}/100) "
                    f"are in a statistical dead heat. Choose based on exclusive tech strengths: "
                    f"**{disp_a.name}** brings exclusive capability in *{', '.join([skills_dict_a[k] for k in list(a_only_keys)[:3]]) or 'core background'}*, "
                    f"while **{disp_b.name}** brings exclusive strength in *{', '.join([skills_dict_b[k] for k in list(b_only_keys)[:3]]) or 'general experience'}*."
                )
            elif ev_a.overall_score > ev_b.overall_score:
                takeaway = (
                    f"**Primary Recommendation: {disp_a.name}** holds a **+{delta_overall} point advantage** overall. "
                    f"Candidate A offers higher technical coverage ({ev_a.technical_score}% vs {ev_b.technical_score}%) "
                    f"and matches {len(a_only_keys)} additional required skill(s) not possessed by Candidate B."
                )
            else:
                takeaway = (
                    f"**Primary Recommendation: {disp_b.name}** holds a **+{abs(delta_overall)} point advantage** overall. "
                    f"Candidate B offers higher technical coverage ({ev_b.technical_score}% vs {ev_a.technical_score}%) "
                    f"and matches {len(b_only_keys)} additional required skill(s) not possessed by Candidate A."
                )

            st.success(takeaway)


# ==============================================================================
# TAB 4: INTERACTIVE RAG RESUME CHAT & READER
# ==============================================================================
with tabs[3]:
    st.subheader("Conversational Resume Reader Agent")
    st.caption("Ask questions across the entire talent pool or deep-dive into a specific candidate with source citations.")

    if not st.session_state.candidates:
        st.info("Please load or upload candidate resumes first to chat with the resume reader.")
    else:
        col_scope, col_clear_chat = st.columns([3, 1])
        scope_options = ["🌐 All Candidates in Talent Pool"] + [f"👤 {c.name}" for c in st.session_state.candidates.values()]
        selected_scope = col_scope.selectbox("Chat Scope", options=scope_options)

        if col_clear_chat.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Determine focused candidate ID if any
        focused_cid = None
        if selected_scope != "🌐 All Candidates in Talent Pool":
            target_name = selected_scope.replace("👤 ", "")
            for cid, c in st.session_state.candidates.items():
                if c.name == target_name:
                    focused_cid = cid
                    break

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("📌 Source Citations & Grounding", expanded=False):
                        for c in msg["citations"]:
                            st.markdown(f"- **{c['candidate_name']}** (*{c['section_type']}*): \"{c['snippet']}\"")

        # Quick Suggested Prompts
        st.markdown("**Suggested Quick Inquiries:**")
        sp1, sp2, sp3 = st.columns(3)
        sample_query = None
        if sp1.button("🔍 'Who has Kubernetes & AWS experience?'"):
            sample_query = "Who has Kubernetes and AWS cloud infrastructure experience?"
        if sp2.button("⚖️ 'Compare top 2 backend candidates'"):
            sample_query = "Compare the top candidates on backend and distributed systems experience."
        if sp3.button("🎓 'Who holds a Master's degree in AI?'"):
            sample_query = "Which candidates hold a Master's degree in AI or Computer Science?"

        # Chat Input
        user_input = st.chat_input("Ask anything about candidates, skills, work history, or qualifications...")
        query_to_run = user_input or sample_query

        if query_to_run:
            st.session_state.chat_history.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            with st.chat_message("assistant"):
                with st.spinner("Searching semantic index and synthesizing grounded answer..."):
                    chat_agent = ResumeChatAgent(
                        vector_store=st.session_state.vector_store,
                        candidates=st.session_state.candidates,
                        llm_client=st.session_state.llm_client
                    )
                    response = chat_agent.chat(
                        query=query_to_run,
                        focused_candidate_id=focused_cid,
                        chat_history=st.session_state.chat_history
                    )

                    st.markdown(response.answer)
                    if response.citations:
                        with st.expander("📌 Source Citations & Grounding", expanded=False):
                            for c in response.citations:
                                st.markdown(f"- **{c.candidate_name}** (*{c.section_type}*): \"{c.snippet}\"")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.answer,
                        "citations": [c.model_dump() for c in response.citations]
                    })
