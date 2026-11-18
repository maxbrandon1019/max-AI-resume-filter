"""Multi-format resume parser supporting PDF, DOCX, and TXT files with PyMuPDF layout-aware extraction and Gemini OCR fallback."""
import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Union, BinaryIO, List, Tuple, Dict, Any, Optional


class ResumeParser:
    """High-accuracy multi-engine resume parser with PyMuPDF layout analysis, table extraction, and fallback mechanisms."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def parse_file(
        cls,
        file_source: Union[str, Path, BinaryIO],
        filename: str = "",
        prefer_multimodal: bool = False,
        engine: str = "auto"
    ) -> str:
        """
        Parses a resume file from a file path or file-like binary stream.
        
        Args:
            file_source: Path to file or open file-like object (bytes)
            filename: Original filename if file_source is a stream
            prefer_multimodal: If True and Gemini API is configured, use Gemini native PDF understanding.
            engine: Parser engine choice ('auto', 'fitz', 'gemini', 'pdfplumber', 'pypdf')
            
        Returns:
            Extracted clean plain text
        """
        text, _ = cls.parse_file_with_metadata(file_source, filename, prefer_multimodal, engine)
        return text

    @classmethod
    def parse_file_with_metadata(
        cls,
        file_source: Union[str, Path, BinaryIO],
        filename: str = "",
        prefer_multimodal: bool = False,
        engine: str = "auto"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parses resume file and returns clean extracted text along with layout and diagnostic metadata.
        
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            filename = path.name
            ext = path.suffix.lower()
            with open(path, "rb") as f:
                content_bytes = f.read()
        else:
            ext = Path(filename).suffix.lower() if filename else ".txt"
            if hasattr(file_source, "read"):
                content_bytes = file_source.read()
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
            else:
                content_bytes = bytes(file_source)

        metadata: Dict[str, Any] = {
            "filename": filename,
            "extension": ext,
            "layout": "single-column",
            "columns_detected": 1,
            "engine": "standard",
            "pages": 1,
            "word_count": 0,
            "is_scanned": False
        }

        if ext == ".pdf":
            text, pdf_meta = cls._parse_pdf_layout_aware(content_bytes, prefer_multimodal, engine)
            metadata.update(pdf_meta)
            cleaned_text = cls.clean_resume_text(text)
            metadata["word_count"] = len(cleaned_text.split())
            return cleaned_text, metadata
        elif ext == ".docx":
            raw_text = cls._parse_docx(content_bytes)
            cleaned_text = cls.clean_resume_text(raw_text)
            metadata["word_count"] = len(cleaned_text.split())
            return cleaned_text, metadata
        elif ext in {".txt", ".md"}:
            raw_text = cls._parse_txt(content_bytes)
            cleaned_text = cls.clean_resume_text(raw_text)
            metadata["word_count"] = len(cleaned_text.split())
            return cleaned_text, metadata
        else:
            raw_text = cls._parse_txt(content_bytes)
            cleaned_text = cls.clean_resume_text(raw_text)
            metadata["word_count"] = len(cleaned_text.split())
            return cleaned_text, metadata

    @classmethod
    def clean_resume_text(cls, text: str) -> str:
        """
        Cleans and normalizes raw extracted text:
        - Resolves ligatures (fi, fl, ffi, etc.) via NFKC normalization
        - Normalizes irregular bullet glyphs (Wingdings, PUA glyphs) to standard '• '
        - De-hyphenates split words across line breaks (e.g. 'implemen-\nted' -> 'implemented')
        - Removes non-breaking spaces, zero-width characters, and excess blank lines
        - Fixes spaced-out uppercase headings (e.g. 'E X P E R I E N C E' -> 'EXPERIENCE')
        """
        if not text:
            return ""

        # 1. NFKC Unicode normalization (resolves ligatures like \ufb01 -> fi, \ufb02 -> fl)
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Standardize bullet symbols
        normalized = re.sub(r"[\uf0b7\uf0a7\u25cf\u25aa\u2022\u2023\u2043\x07\u25cb\u25e6]", "• ", normalized)

        # 3. Clean zero-width, soft hyphens, and non-breaking spaces
        normalized = normalized.replace("\u200b", "").replace("\ufeff", "").replace("\xad", "")
        normalized = normalized.replace("\xa0", " ")

        # 4. De-hyphenate words broken across line breaks
        normalized = re.sub(r"(\b[a-zA-Z]{3,})-\n\s*([a-zA-Z]{3,}\b)", r"\1\2", normalized)

        # 5. Fix spaced-out headings (e.g. 'S K I L L S' -> 'SKILLS')
        def _collapse_spaced_header(match):
            word = match.group(0)
            collapsed = word.replace(" ", "")
            if collapsed in {"EXPERIENCE", "EDUCATION", "SKILLS", "SUMMARY", "PROJECTS", "CERTIFICATIONS"}:
                return collapsed
            return word

        normalized = re.sub(r"\b[A-Z](?:\s+[A-Z]){3,15}\b", _collapse_spaced_header, normalized)

        # 6. Ensure clean spacing around bullets
        normalized = re.sub(r"\n\s*•\s*", "\n• ", normalized)

        # 7. Collapse excessive consecutive newlines (keep max 2 for paragraph breaks)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()

    @classmethod
    def _parse_pdf_layout_aware(
        cls,
        data: bytes,
        prefer_multimodal: bool = False,
        engine: str = "auto"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Tiered high-accuracy PDF parsing pipeline:
        Tier 1: PyMuPDF (fitz) smart block/line clustering with gutter analysis & table extraction.
        Tier 2: Gemini Multimodal Document Vision (if requested or scanned).
        Tier 3: pdfplumber spatial extraction.
        Tier 4: pypdf fallback.
        """
        meta: Dict[str, Any] = {
            "layout": "single-column",
            "columns_detected": 1,
            "engine": "unknown",
            "pages": 1,
            "tables_extracted": 0,
            "is_scanned": False
        }

        # Force Gemini Multimodal if requested
        if (prefer_multimodal or engine == "gemini") and os.getenv("GEMINI_API_KEY"):
            gemini_text = cls._parse_pdf_with_gemini(data)
            if gemini_text:
                meta["engine"] = "gemini-multimodal"
                return gemini_text, meta

        # Force pdfplumber if explicitly chosen
        if engine == "pdfplumber":
            return cls._parse_pdf_pdfplumber(data)

        # Primary: PyMuPDF (fitz)
        if engine in {"auto", "fitz"}:
            fitz_result = cls._parse_pdf_fitz(data)
            if fitz_result:
                text, fitz_meta = fitz_result
                # Check for scanned PDF (< 35 words across document)
                if fitz_meta.get("is_scanned", False) and os.getenv("GEMINI_API_KEY"):
                    gemini_text = cls._parse_pdf_with_gemini(data)
                    if gemini_text:
                        fitz_meta["engine"] = "gemini-multimodal-ocr"
                        fitz_meta["is_scanned"] = False
                        return gemini_text, fitz_meta
                if text.strip():
                    return text, fitz_meta

        # Secondary fallback: pdfplumber
        plumber_result = cls._parse_pdf_pdfplumber(data)
        if plumber_result and plumber_result[0].strip():
            return plumber_result

        # Tertiary fallback: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_pages.append(f"--- [Page {i+1}] ---\n{page_text.strip()}")
            if text_pages:
                meta["engine"] = "pypdf"
                meta["pages"] = len(reader.pages)
                return "\n\n".join(text_pages), meta
        except Exception:
            pass

        # Final raw string fallback
        try:
            decoded = data.decode("utf-8", errors="ignore")
            lines = [line.strip() for line in decoded.splitlines() if any(c.isalnum() for c in line)]
            meta["engine"] = "raw-fallback"
            return "\n".join(lines), meta
        except Exception:
            return "", meta

    @classmethod
    def _parse_pdf_fitz(cls, data: bytes) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extracts PDF text using PyMuPDF (fitz) with layout-aware spatial block/line ordering
        and table extraction to prevent column interleaving.
        """
        try:
            import fitz
        except ImportError:
            return None

        try:
            doc = fitz.open(stream=data, filetype="pdf")
            if len(doc) == 0:
                return "", {"layout": "single-column", "pages": 0, "engine": "fitz-empty"}

            pages_text: List[str] = []
            multi_col_detected = False
            total_tables = 0
            total_words = 0

            for i, page in enumerate(doc):
                # 1. Extract tables cleanly if present
                table_texts = []
                if hasattr(page, "find_tables"):
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            for tab in tabs.tables:
                                extracted_tab = tab.extract()
                                if extracted_tab:
                                    rows = []
                                    for row in extracted_tab:
                                        cells = [str(c).strip().replace("\n", " ") for c in row if c is not None and str(c).strip()]
                                        if cells:
                                            rows.append(" | ".join(cells))
                                    if rows:
                                        table_texts.append("\n".join(rows))
                                        total_tables += 1
                    except Exception:
                        pass

                # 2. Extract lines with bounding boxes
                page_text, is_multi = cls._extract_fitz_page_lines(page)
                if is_multi:
                    multi_col_detected = True

                combined_page = page_text
                if table_texts:
                    combined_page = combined_page + "\n\n" + "\n\n".join(table_texts)

                if combined_page.strip():
                    pages_text.append(f"--- [Page {i+1}] ---\n{combined_page.strip()}")
                    total_words += len(combined_page.split())

            full_text = "\n\n".join(pages_text)
            is_scanned = (total_words < 35 and len(doc) >= 1)

            meta = {
                "layout": "multi-column" if multi_col_detected else "single-column",
                "columns_detected": 2 if multi_col_detected else 1,
                "engine": "pymupdf-smart-layout",
                "pages": len(doc),
                "tables_extracted": total_tables,
                "is_scanned": is_scanned
            }
            return full_text, meta
        except Exception:
            return None

    @classmethod
    def _extract_fitz_page_lines(cls, page) -> Tuple[str, bool]:
        """
        Extracts lines from a PyMuPDF page, detects vertical column gutters,
        and sequences text so left and right columns are never mixed horizontally.
        """
        data_dict = page.get_text("dict")
        width = float(page.rect.width)
        height = float(page.rect.height)

        extracted_lines: List[Dict[str, Any]] = []
        for block in data_dict.get("blocks", []):
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if text:
                        bbox = line.get("bbox", (0, 0, 0, 0))
                        extracted_lines.append({
                            "x0": float(bbox[0]),
                            "y0": float(bbox[1]),
                            "x1": float(bbox[2]),
                            "y1": float(bbox[3]),
                            "text": text
                        })

        if not extracted_lines:
            return page.get_text() or "", False

        # 1. Search for candidate vertical column gutters between 20% and 75% of page width
        best_gutter = None
        min_crossing = float("inf")
        step = 5.0
        x = width * 0.20
        total_lines = len(extracted_lines)

        while x <= width * 0.75:
            # Lines crossing the vertical test line (excluding full-width header lines)
            crossing = [
                l for l in extracted_lines
                if l["x0"] < x and l["x1"] > x and (l["x1"] - l["x0"]) < width * 0.45
            ]
            left = [l for l in extracted_lines if l["x1"] <= x]
            right = [l for l in extracted_lines if l["x0"] >= x]

            if len(left) >= total_lines * 0.12 and len(right) >= total_lines * 0.18:
                if len(crossing) < min_crossing:
                    min_crossing = len(crossing)
                    best_gutter = x
            x += step

        # 2. If a clean gutter with low crossings is found, partition columns
        if best_gutter is not None and min_crossing <= 3:
            left_lines = [l for l in extracted_lines if l["x1"] <= best_gutter + 5.0]
            right_lines = [l for l in extracted_lines if l["x0"] >= best_gutter - 5.0]

            min_left_y = min(l["y0"] for l in left_lines)
            min_right_y = min(l["y0"] for l in right_lines)
            first_col_y = min(min_left_y, min_right_y)
            max_col_y = max(max(l["y1"] for l in left_lines), max(l["y1"] for l in right_lines))

            headers: List[Dict[str, Any]] = []
            col1: List[Dict[str, Any]] = []
            col2: List[Dict[str, Any]] = []
            footers: List[Dict[str, Any]] = []

            for l in extracted_lines:
                # Text above both columns
                if l["y1"] <= first_col_y:
                    headers.append(l)
                elif l["x0"] < best_gutter and l["x1"] > best_gutter:
                    # Spanning line
                    if l["y0"] < first_col_y + 30:
                        headers.append(l)
                    elif l["y0"] > max_col_y - 20:
                        footers.append(l)
                    else:
                        col2.append(l)
                elif l["x1"] <= best_gutter + 5.0:
                    col1.append(l)
                else:
                    col2.append(l)

            headers_text = "\n".join(l["text"] for l in sorted(headers, key=lambda l: l["y0"]))
            col1_text = "\n".join(l["text"] for l in sorted(col1, key=lambda l: l["y0"]))
            col2_text = "\n".join(l["text"] for l in sorted(col2, key=lambda l: l["y0"]))
            footers_text = "\n".join(l["text"] for l in sorted(footers, key=lambda l: l["y0"]))

            parts = [p for p in [headers_text, col1_text, col2_text, footers_text] if p.strip()]
            return "\n\n".join(parts), True

        # Single column reading order: sort lines by rounded vertical coordinate, then horizontal
        sorted_lines = sorted(extracted_lines, key=lambda l: (round(l["y0"] / 3.0), l["x0"]))
        return "\n".join(l["text"] for l in sorted_lines), False

    # Alias for test backwards-compatibility
    _extract_page_fitz = _extract_fitz_page_lines

    @classmethod
    def _parse_pdf_pdfplumber(cls, data: bytes) -> Tuple[str, Dict[str, Any]]:
        """Fallback pdfplumber parser with spatial column sorting."""
        meta = {"layout": "single-column", "columns_detected": 1, "engine": "pdfplumber"}
        try:
            import pdfplumber
            text_pages: List[str] = []
            multi_col_detected = False

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text, is_multi = cls._extract_page_with_column_detection(page)
                    if is_multi:
                        multi_col_detected = True
                    if page_text.strip():
                        text_pages.append(f"--- [Page {i+1}] ---\n{page_text.strip()}")

            if text_pages:
                meta["layout"] = "multi-column" if multi_col_detected else "single-column"
                meta["columns_detected"] = 2 if multi_col_detected else 1
                meta["engine"] = "pdfplumber-layout"
                meta["pages"] = len(text_pages)
                return "\n\n".join(text_pages), meta
        except Exception:
            pass
        return "", meta

    @classmethod
    def _extract_page_with_column_detection(cls, page) -> Tuple[str, bool]:
        """pdfplumber column detection method for backward compatibility."""
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        if not words:
            return page.extract_text() or "", False

        width = float(page.width)
        height = float(page.height)

        best_split = None
        min_crossing_words = float("inf")
        step = 10.0
        start_x = width * 0.22
        end_x = width * 0.75

        curr_x = start_x
        while curr_x <= end_x:
            crossing = 0
            left_count = 0
            right_count = 0

            for w in words:
                x0 = float(w["x0"])
                x1 = float(w["x1"])
                top = float(w["top"])

                if top < height * 0.15:
                    continue

                if x0 < curr_x and x1 > curr_x:
                    crossing += 1
                elif x1 <= curr_x:
                    left_count += 1
                else:
                    right_count += 1

            total_body_words = max(1, left_count + right_count + crossing)
            if left_count / total_body_words >= 0.15 and right_count / total_body_words >= 0.20:
                if crossing < min_crossing_words:
                    min_crossing_words = crossing
                    best_split = curr_x

            curr_x += step

        if best_split is not None and min_crossing_words <= 4:
            header_words = []
            left_words = []
            right_words = []

            for w in words:
                top = float(w["top"])
                x0 = float(w["x0"])
                x1 = float(w["x1"])

                if top < height * 0.15 and (x0 < best_split and x1 > best_split or (x1 - x0) > width * 0.5):
                    header_words.append(w)
                elif x1 <= best_split + 4.0:
                    left_words.append(w)
                else:
                    right_words.append(w)

            header_text = cls._reconstruct_text_from_words(header_words)
            left_text = cls._reconstruct_text_from_words(left_words)
            right_text = cls._reconstruct_text_from_words(right_words)

            sections = []
            if header_text.strip():
                sections.append(header_text.strip())
            if left_text.strip():
                sections.append(left_text.strip())
            if right_text.strip():
                sections.append(right_text.strip())

            return "\n\n".join(sections), True

        return page.extract_text() or "", False

    @classmethod
    def _reconstruct_text_from_words(cls, words: List[Dict[str, Any]], y_tolerance: float = 3.0) -> str:
        """Groups words into horizontal lines and sorts them top-to-bottom, left-to-right."""
        if not words:
            return ""

        sorted_words = sorted(words, key=lambda w: (round(float(w["top"]) / y_tolerance), float(w["x0"])))

        lines: List[List[str]] = []
        curr_line: List[str] = []
        curr_y = None

        for w in sorted_words:
            top = float(w["top"])
            if curr_y is None or abs(top - curr_y) <= y_tolerance:
                curr_line.append(w["text"])
                curr_y = top
            else:
                if curr_line:
                    lines.append(" ".join(curr_line))
                curr_line = [w["text"]]
                curr_y = top

        if curr_line:
            lines.append(" ".join(curr_line))

        return "\n".join(lines)

    @classmethod
    def _parse_pdf_with_gemini(cls, data: bytes) -> Optional[str]:
        """Uses Gemini multimodal document vision to transcribe PDFs with full layout fidelity."""
        try:
            import google.generativeai as genai
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are an expert resume parsing engine. Transcribe this resume faithfully into clean, structured Markdown.\n"
                "Guidelines:\n"
                "1. Preserve all sections, headings, dates, company names, job titles, education, and bullet points.\n"
                "2. If the resume has multiple columns, sidebars, or tables, transcribe each section sequentially as a complete unit (e.g. Header -> Summary -> Skills -> Experience -> Education).\n"
                "3. Do NOT interleave words or sentences between adjacent columns.\n"
                "4. Normalize all bullet points to standard '• '.\n"
                "5. Maintain exact numbers, metrics, and technology names."
            )
            response = model.generate_content([
                {"mime_type": "application/pdf", "data": data},
                prompt
            ])
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass
        return None

    @classmethod
    def _parse_docx(cls, data: bytes) -> str:
        """Parses DOCX bytes using python-docx or docx2txt."""
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Also extract text inside tables cleanly
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
                        
            return "\n\n".join(paragraphs)
        except ImportError:
            pass
        except Exception:
            pass

        return cls._parse_txt(data)

    @classmethod
    def _parse_txt(cls, data: bytes) -> str:
        """Parses plain text bytes with multiple encoding fallbacks."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return data.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore").strip()
