"""Executive Report Generator for Hiring Managers and Recruiter ATS Export.

Generates:
1. Standalone, beautifully styled printable HTML/PDF Executive Briefs.
2. Clean Markdown Dossiers for Notion / Slack / ATS.
3. CSV Leaderboard summaries for candidate pool export.
"""
import csv
import io
from typing import Dict, List, Optional
from datetime import datetime

from src.models.schemas import CandidateProfile, FitEvaluation, JDRequirements


class ReportGenerator:
    """Produces executive candidate briefs, interview rubrics, and CSV exports."""

    @classmethod
    def generate_executive_html(
        cls,
        candidate: CandidateProfile,
        evaluation: FitEvaluation,
        jd: Optional[JDRequirements] = None
    ) -> str:
        """
        Generates a standalone, beautifully styled HTML executive report.
        Includes @media print styling for 1-click 'Save as PDF' from browser.
        """
        job_title = jd.job_title if jd else "Target Engineering Role"
        current_date = datetime.now().strftime("%B %d, %Y")

        # Color scheme based on fit tier
        if evaluation.overall_score >= 75.0:
            badge_color = "#059669"
            badge_bg = "#d1fae5"
            border_color = "#10b981"
        elif evaluation.overall_score >= 55.0:
            badge_color = "#2563eb"
            badge_bg = "#dbeafe"
            border_color = "#3b82f6"
        else:
            badge_color = "#dc2626"
            badge_bg = "#fee2e2"
            border_color = "#ef4444"

        # Skills matrix rows
        matched_rows = []
        for sm in evaluation.skills_matrix:
            if sm.matched:
                req_tag = '<span class="tag-req">Required</span>' if sm.is_required else '<span class="tag-pref">Preferred</span>'
                evidence = sm.candidate_evidence or "Confirmed in resume"
                matched_rows.append(f"""
                <tr>
                    <td><strong>{sm.skill_name}</strong></td>
                    <td>{req_tag}</td>
                    <td><span class="status-match">✓ Matched</span></td>
                    <td class="evidence-cell">"{evidence}"</td>
                </tr>
                """)

        missing_rows = []
        for sm in evaluation.skills_matrix:
            if not sm.matched:
                req_tag = '<span class="tag-req">Required</span>' if sm.is_required else '<span class="tag-pref">Preferred</span>'
                missing_rows.append(f"""
                <tr>
                    <td><strong>{sm.skill_name}</strong></td>
                    <td>{req_tag}</td>
                    <td><span class="status-missing">✕ Not Found</span></td>
                    <td class="evidence-cell">No explicit evidence found in resume sections</td>
                </tr>
                """)

        # Strengths & Gaps lists
        strengths_html = "".join([f"<li><strong>Strengths:</strong> {s}</li>" for s in evaluation.strengths])
        gaps_html = "".join([f"<li><strong>Area of Concern:</strong> {g}</li>" for g in evaluation.gaps])

        # Interview questions rubric
        interview_rows = []
        for idx, q in enumerate(evaluation.tailored_interview_questions, 1):
            interview_rows.append(f"""
            <div class="interview-card">
                <div class="q-header">
                    <span class="q-num">Q{idx}</span>
                    <span class="q-text">{q}</span>
                </div>
                <div class="rating-scale">
                    <span>Evaluation: </span>
                    <label><input type="radio" name="q{idx}" disabled> 1 - Unsatisfactory</label>
                    <label><input type="radio" name="q{idx}" disabled> 2 - Developing</label>
                    <label><input type="radio" name="q{idx}" disabled> 3 - Proficient</label>
                    <label><input type="radio" name="q{idx}" disabled> 4 - Exceptional</label>
                </div>
                <div class="notes-box">
                    <em>Interviewer Notes:</em>
                    <div class="notes-lines"></div>
                </div>
            </div>
            """)
        interview_section = "".join(interview_rows)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Executive Candidate Brief - {evaluation.candidate_name}</title>
<style>
    @page {{
        size: A4 portrait;
        margin: 0.6in 0.6in;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        line-height: 1.45;
        background-color: #f8fafc;
        margin: 0;
        padding: 20px;
    }}
    .container {{
        max-width: 850px;
        margin: 0 auto;
        background: #ffffff;
        padding: 36px 44px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }}
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 20px;
        margin-bottom: 24px;
    }}
    .header-left h1 {{
        margin: 0 0 6px 0;
        font-size: 26px;
        color: #0f172a;
    }}
    .header-left .role-sub {{
        font-size: 15px;
        color: #64748b;
        font-weight: 500;
    }}
    .header-right {{
        text-align: right;
    }}
    .fit-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 15px;
        color: {badge_color};
        background: {badge_bg};
        border: 1px solid {border_color};
    }}
    .date-str {{
        font-size: 12px;
        color: #94a3b8;
        margin-top: 6px;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }}
    .metric-box {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }}
    .metric-val {{
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
    }}
    .metric-lbl {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-top: 4px;
    }}
    .summary-box {{
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 14.5px;
        color: #334155;
        margin-bottom: 24px;
    }}
    h2 {{
        font-size: 17px;
        color: #0f172a;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        margin-bottom: 20px;
    }}
    .card-list {{
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 18px;
    }}
    .card-list ul {{
        margin: 0;
        padding-left: 18px;
    }}
    .card-list li {{
        margin-bottom: 8px;
        font-size: 13.5px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 20px;
    }}
    th, td {{
        padding: 8px 10px;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
    }}
    th {{
        background: #f8fafc;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
    }}
    .status-match {{
        color: #059669;
        font-weight: 600;
    }}
    .status-missing {{
        color: #dc2626;
        font-weight: 600;
    }}
    .tag-req {{
        background: #e0e7ff;
        color: #3730a3;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }}
    .tag-pref {{
        background: #f1f5f9;
        color: #475569;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
    }}
    .evidence-cell {{
        color: #475569;
        font-style: italic;
        font-size: 12px;
    }}
    .interview-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
        page-break-inside: avoid;
    }}
    .q-header {{
        display: flex;
        align-items: baseline;
        gap: 8px;
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 8px;
    }}
    .q-num {{
        background: #3b82f6;
        color: #ffffff;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 11px;
    }}
    .rating-scale {{
        font-size: 12px;
        color: #64748b;
        margin-bottom: 10px;
        display: flex;
        gap: 16px;
    }}
    .notes-box {{
        font-size: 12px;
        color: #64748b;
    }}
    .notes-lines {{
        height: 36px;
        border-bottom: 1px dashed #cbd5e1;
        margin-top: 4px;
    }}
    .print-btn {{
        display: block;
        width: 100%;
        max-width: 260px;
        margin: 20px auto 0 auto;
        padding: 10px 16px;
        background: #0f172a;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
    }}
    .print-btn:hover {{
        background: #334155;
    }}
    @media print {{
        body {{
            background: #ffffff;
            padding: 0;
        }}
        .container {{
            box-shadow: none;
            padding: 0;
        }}
        .print-btn {{
            display: none;
        }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header-bar">
        <div class="header-left">
            <h1>{evaluation.candidate_name}</h1>
            <div class="role-sub">Role Assessment: <strong>{job_title}</strong></div>
            <div class="role-sub">Experience: {candidate.total_years_experience} Years • Education: {candidate.highest_education}</div>
        </div>
        <div class="header-right">
            <div class="fit-badge">{evaluation.fit_tier} • {evaluation.overall_score}/100</div>
            <div class="date-str">Report Date: {current_date}</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-val">{evaluation.overall_score}</div>
            <div class="metric-lbl">Overall Fit</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{evaluation.technical_score}%</div>
            <div class="metric-lbl">Technical Match</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{evaluation.experience_score}%</div>
            <div class="metric-lbl">Experience Alignment</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{evaluation.matched_skills_count}/{evaluation.total_required_skills_count}</div>
            <div class="metric-lbl">Skills Matched</div>
        </div>
    </div>

    <div class="summary-box">
        <strong>Executive Assessment:</strong> {evaluation.fit_summary}
    </div>

    <div class="two-col">
        <div class="card-list" style="border-left: 4px solid #10b981;">
            <h3 style="margin-top:0; font-size:14px; color:#065f46;">🟢 Key Strengths</h3>
            <ul>
                {strengths_html if strengths_html else "<li>Aligned technical background with required stack.</li>"}
            </ul>
        </div>
        <div class="card-list" style="border-left: 4px solid #f59e0b;">
            <h3 style="margin-top:0; font-size:14px; color:#92400e;">⚠️ Gaps & Areas to Probe</h3>
            <ul>
                {gaps_html if gaps_html else "<li>No critical gaps identified against requirements.</li>"}
            </ul>
        </div>
    </div>

    <h2>🧩 Skills Match Matrix</h2>
    <table>
        <thead>
            <tr>
                <th>Skill</th>
                <th>Importance</th>
                <th>Match Status</th>
                <th>Candidate Evidence Snippet</th>
            </tr>
        </thead>
        <tbody>
            {"".join(matched_rows)}
            {"".join(missing_rows)}
        </tbody>
    </table>

    <h2>🎯 Targeted Interview Assessment Rubric</h2>
    <p style="font-size:12.5px; color:#64748b; margin-top:-6px; margin-bottom:14px;">
        Structured interview questions formulated based on this candidate's specific qualification gaps and strengths:
    </p>
    {interview_section}

    <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
</div>
</body>
</html>
"""
        return html

    @classmethod
    def generate_markdown_dossier(
        cls,
        candidate: CandidateProfile,
        evaluation: FitEvaluation,
        jd: Optional[JDRequirements] = None
    ) -> str:
        """Generates clean GitHub-Flavored Markdown candidate report."""
        job_title = jd.job_title if jd else "Target Engineering Role"
        current_date = datetime.now().strftime("%Y-%m-%d")

        lines = [
            f"# Candidate Evaluation Dossier: {evaluation.candidate_name}",
            f"**Target Role:** {job_title}  ",
            f"**Assessment Date:** {current_date}  ",
            f"**Fit Status:** {evaluation.fit_tier} ({evaluation.overall_score}/100)  ",
            f"**Contact Details:** Email: {candidate.email or 'N/A'} | Phone: {candidate.phone or 'N/A'} | Location: {candidate.location or 'N/A'}  ",
            f"**Detected Seniority:** {candidate.total_years_experience} Years | **Education:** {candidate.highest_education}",
            "",
            "---",
            "",
            "## 📊 Score Breakdown",
            f"- **Overall Fit Score:** `{evaluation.overall_score} / 100`",
            f"- **Technical Skills Match:** `{evaluation.technical_score}%`",
            f"- **Experience & Seniority Fit:** `{evaluation.experience_score}%`",
            f"- **Academic Fit:** `{evaluation.education_score}%`",
            f"- **Core Skills Verified:** `{evaluation.matched_skills_count} / {evaluation.total_required_skills_count}`",
            "",
            "## 💡 Executive Summary",
            f"> {evaluation.fit_summary}",
            "",
            "## 🟢 Candidate Strengths",
        ]

        for s in evaluation.strengths:
            lines.append(f"- {s}")

        lines.extend(["", "## ⚠️ Gaps & Areas to Probe"])
        for g in evaluation.gaps:
            lines.append(f"- {g}")

        lines.extend([
            "",
            "## 🧩 Detailed Skill Matrix",
            "| Skill | Requirement | Status | Candidate Evidence |",
            "| :--- | :--- | :--- | :--- |"
        ])

        for sm in evaluation.skills_matrix:
            req_str = "Required" if sm.is_required else "Preferred"
            status_str = "✅ Matched" if sm.matched else "❌ Missing"
            evidence_str = (sm.candidate_evidence or "N/A").replace("|", "/")
            lines.append(f"| **{sm.skill_name}** | {req_str} | {status_str} | {evidence_str} |")

        lines.extend([
            "",
            "## 🎯 Tailored Interview Questions",
            "Use these prompts during the technical and behavioral screening loops:"
        ])

        for idx, q in enumerate(evaluation.tailored_interview_questions, 1):
            lines.append(f"{idx}. {q}")

        return "\n".join(lines)

    @classmethod
    def generate_leaderboard_csv(
        cls,
        evaluations: List[FitEvaluation],
        candidates: Dict[str, CandidateProfile]
    ) -> str:
        """Exports full candidate leaderboard ranking to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Rank",
            "Candidate ID",
            "Candidate Name",
            "Email",
            "Phone",
            "Location",
            "Fit Tier",
            "Overall Score",
            "Technical Score",
            "Experience Score",
            "Education Score",
            "Skills Matched",
            "Total Required Skills",
            "Years Experience",
            "Highest Education",
            "Core Skills",
            "Top Strengths",
            "Key Gaps",
            "Filename"
        ])

        # Sort descending by overall score
        sorted_evals = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)

        for rank, ev in enumerate(sorted_evals, 1):
            cand = candidates.get(ev.candidate_id)
            exp = cand.total_years_experience if cand else 0.0
            edu = cand.highest_education if cand else "N/A"
            filename = cand.filename if cand else "N/A"
            email = (cand.email or "—") if cand and not getattr(cand, "is_anonymized", False) else "—"
            phone = (cand.phone or "—") if cand and not getattr(cand, "is_anonymized", False) else "—"
            loc = (cand.location or "—") if cand else "—"
            skills_str = ", ".join(cand.top_skills) if cand and cand.top_skills else "—"
            strengths_str = "; ".join(ev.strengths)
            gaps_str = "; ".join(ev.gaps)

            writer.writerow([
                rank,
                ev.candidate_id,
                ev.candidate_name,
                email,
                phone,
                loc,
                ev.fit_tier,
                ev.overall_score,
                ev.technical_score,
                ev.experience_score,
                ev.education_score,
                ev.matched_skills_count,
                ev.total_required_skills_count,
                exp,
                edu,
                skills_str,
                strengths_str,
                gaps_str,
                filename
            ])

        return output.getvalue()
