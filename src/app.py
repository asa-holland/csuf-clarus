from flask import Flask, request, render_template, jsonify, send_file
import os
import sys
import logging
import math
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime
from clarus.steps.preprocess import extract_text_from_file, allowed_file
from clarus.steps.corpus_operations import create_corpus_zip, preprocess_corpus_files
from clarus.analysis.document_processor import DocumentProcessor
from clarus.analysis.semantic_extractor import SemanticExtractor
from clarus.analysis.terminology_validator import TerminologyValidator
from clarus.analysis.contradiction_analyzer import ContradictionDetector
import numpy as np
import re
import nltk
import os
from typing import Dict, List, Any, Optional, Union, Tuple


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
)
logger = logging.getLogger("clarus.app")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(_handler)
logger.propagate = False

nltk_data_path = os.path.expanduser("~/nltk_data")
if os.path.exists("/root/nltk_data"):
    nltk.data.path.append("/root/nltk_data")
elif os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)

os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/transformers"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def find_term_context(term: str, text: str, window_size: int = 1) -> str:
    from nltk.tokenize import sent_tokenize

    if not term or not text:
        return "Context not available"

    pattern = re.compile(re.escape(term), re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if not matches:
        return "Term not found in text"

    match = matches[0]
    start, end = match.span()

    sentences = sent_tokenize(text)

    current_pos = 0
    target_sentence_idx = -1

    for i, sent in enumerate(sentences):
        sent_start = text.find(sent, current_pos)
        sent_end = sent_start + len(sent)
        current_pos = sent_end

        if sent_start <= start < sent_end:
            target_sentence_idx = i
            break

    if target_sentence_idx == -1:
        return "Context not available"

    start_idx = max(0, target_sentence_idx - window_size)
    end_idx = min(len(sentences), target_sentence_idx + window_size + 1)
    context_sentences = sentences[start_idx:end_idx]

    target_sentence = context_sentences[target_sentence_idx - start_idx]
    highlighted_sentence = pattern.sub(
        f"<strong>{match.group(0)}</strong>", target_sentence
    )
    context_sentences[target_sentence_idx - start_idx] = highlighted_sentence

    return " ... ".join(context_sentences)


def _safe_float(v: float) -> object:
    if math.isnan(v) or math.isinf(v):
        logger.warning("Non-finite float detected (%s) — serialising as null", v)
        return None
    return v


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return _safe_float(float(obj))
    elif isinstance(obj, float):
        return _safe_float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


template_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "clarus", "templates"
)
static_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "clarus", "static"
)

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
@app.route("/pipeline")
def index():
    return render_template("pipeline.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": "File type not supported. Allowed types: PDF, DOCX, HTML, TXT"
                    }
                ),
                400,
            )

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        file_extension = filename.rsplit(".", 1)[1].lower()
        text, metadata = extract_text_from_file(file_path, file_extension)

        metadata.update(
            {
                "original_filename": file.filename,
                "file_size": os.path.getsize(file_path),
                "processed_at": datetime.now().isoformat(),
                "text_length": len(text),
            }
        )

        os.remove(file_path)

        return jsonify({"success": True, "text": text, "metadata": metadata})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/download-corpus")
def download_corpus():
    try:
        zip_path = Path("uploads/corpus_files.zip")
        create_corpus_zip("corpus_files", zip_path)
        return send_file(zip_path, as_attachment=True, download_name="corpus_files.zip")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/preprocess-corpus", methods=["GET", "POST"])
def preprocess_corpus():
    try:
        result = preprocess_corpus_files("corpus_files", "corpus_files_preprocessed")
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-preprocessed")
def download_preprocessed():
    try:
        output_dir = Path("corpus_files_preprocessed")
        if not output_dir.exists():
            return (
                jsonify(
                    {"error": "Preprocessed corpus not found. Run preprocessing first."}
                ),
                404,
            )

        zip_path = Path("uploads/corpus_files_preprocessed.zip")
        create_corpus_zip("corpus_files_preprocessed", zip_path)
        return send_file(
            zip_path, as_attachment=True, download_name="corpus_files_preprocessed.zip"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process-document", methods=["POST"])
def process_document():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        processor = DocumentProcessor()
        result = processor.process_document(data["text"])

        segments_data = []
        for segment in result.segments:
            segment_data = {
                "text": segment.text,
                "element_type": segment.element_type.value,
                "section_number": segment.section_number,
                "heading": segment.heading,
                "start_line": segment.start_line,
                "end_line": segment.end_line,
                "confidence": segment.confidence,
                "errors": segment.errors,
            }

            if segment.semantic_anchors:
                segment_data["semantic_anchors"] = {
                    "condition": segment.semantic_anchors.condition,
                    "subject": segment.semantic_anchors.subject,
                    "modality": segment.semantic_anchors.modality,
                    "object": segment.semantic_anchors.object,
                    "temporal": segment.semantic_anchors.temporal,
                    "negation": segment.semantic_anchors.negation,
                    "confidence": segment.semantic_anchors.confidence,
                }

            segments_data.append(segment_data)

        return jsonify(
            {
                "success": True,
                "segments": segments_data,
                "processing_stats": result.processing_stats,
                "metadata": result.metadata,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract-semantics", methods=["POST"])
def extract_semantics():
    try:
        data = request.get_json()
        if not data or "segments" not in data:
            return jsonify({"error": "No segments provided"}), 400

        extractor = SemanticExtractor()
        profiles = []
        total_confidence = 0

        for segment_data in data["segments"]:
            profile = extractor.extract_semantic_profile(segment_data["text"])
            profile_data = {
                "original_sentence": profile.original_sentence,
                "confidence": profile.confidence,
                "modality": profile.modality.text if profile.modality else None,
                "condition": profile.condition.text if profile.condition else None,
                "subject": profile.subject.text if profile.subject else None,
                "object": profile.object.text if profile.object else None,
                "temporal": profile.temporal.text if profile.temporal else None,
                "negation": profile.negation.text if profile.negation else None,
            }
            # A profile is only meaningful when it has the four core components.
            # Temporal and negation are optional bonuses.
            required_keys = ("subject", "object", "modality", "condition")
            if all(profile_data[k] for k in required_keys):
                profiles.append(profile_data)
                total_confidence += profile.confidence

        avg_confidence = total_confidence / len(profiles) if profiles else 0

        return jsonify(
            {
                "success": True,
                "profiles": profiles,
                "total_profiles": len(profiles),
                "avg_confidence": avg_confidence,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        terminology_validator = TerminologyValidator()
        contradiction_detector = ContradictionDetector(config={"enable_ml": False})

        extracted_terms = terminology_validator.extract_terms_from_text(data["text"])

        # Use quality-based selection instead of fixed limit
        min_quality_threshold = 0.4
        max_terms = 50

        # Filter terms by quality score and limit
        high_quality_terms = []
        for term in extracted_terms:
            # Get validation result to check quality
            result = terminology_validator.validate_term(term)
            if (
                result.classification.confidence >= min_quality_threshold
                and result.classification.is_domain_term
                and not result.classification.is_common_english
            ):
                high_quality_terms.append(term)
            if len(high_quality_terms) >= max_terms:
                break

        if not high_quality_terms:
            # Fall back only to multi-word terms or capitalised proper nouns —
            # avoids flooding the UI with ordinary single-word game vocabulary.
            high_quality_terms = [
                t for t in extracted_terms
                if " " in t or (t[0].isupper() and not t.isupper())
            ][:max_terms]

        terminology_result = terminology_validator.validate_terms(
            high_quality_terms, full_text=data["text"]
        )

        document_text = data.get("text", "")

        logger.info(
            "analyze-document: %d term(s) extracted, %d high-quality selected",
            len(extracted_terms),
            len(high_quality_terms),
        )
        logger.debug(
            "terminology stats: defined=%d undefined=%d avg_confidence=%s",
            terminology_result.statistics.get("defined_terms"),
            terminology_result.statistics.get("undefined_terms"),
            terminology_result.statistics.get("avg_confidence"),
        )

        def create_term_entry(result):
            raw_confidence = result.classification.confidence
            if math.isnan(raw_confidence) or math.isinf(raw_confidence):
                logger.warning(
                    "Non-finite confidence for term '%s': %s — clamping to 0.0",
                    result.term,
                    raw_confidence,
                )
                raw_confidence = 0.0
            is_undefined_acronym = (
                result.term.isupper()
                and len(result.term) >= 2
                and not result.is_defined
            )
            return {
                "term": result.term,
                "normalized_term": result.normalized_term,
                "is_defined": result.is_defined,
                "is_undefined_acronym": is_undefined_acronym,
                "classification": {
                    "is_domain_term": result.classification.is_domain_term,
                    "is_common_english": result.classification.is_common_english,
                    "confidence": float(raw_confidence),
                    "classification_reason": result.classification.classification_reason,
                },
                "suggested_definition": result.suggested_definition,
                "context_excerpt": result.context_excerpt,
            }

        terminology_data = {
            "statistics": convert_numpy_types(terminology_result.statistics),
            "defined_terms": [
                create_term_entry(result) for result in terminology_result.defined_terms
            ],
            "undefined_terms": [
                create_term_entry(result)
                for result in terminology_result.undefined_terms
            ],
            "term_occurrences": [
                {
                    "term": result.term,
                    "is_defined": result.is_defined,
                    "is_undefined_acronym": (
                        result.term.isupper()
                        and len(result.term) >= 2
                        and not result.is_defined
                    ),
                    "context_excerpt": result.context_excerpt,
                }
                for result in terminology_result.defined_terms
                + terminology_result.undefined_terms
            ],
        }

        candidate_statements = []
        if "segments" in data:
            for segment in data["segments"]:
                if segment.get("element_type") in ("normative", "unknown"):
                    candidate_statements.append(segment["text"])
        else:
            processor = DocumentProcessor()
            processed = processor.process_document(data["text"])
            candidate_statements = [
                seg.text
                for seg in processed.segments
                if seg.element_type.value in ("normative", "unknown")
            ]

        logger.info(
            "contradiction detection: %d candidate statements (%s)",
            len(candidate_statements),
            candidate_statements,
        )
        contradictions = contradiction_detector.detect_contradictions(
            candidate_statements
        )
        logger.info(
            "contradiction detection: %d contradiction(s) found", len(contradictions)
        )

        contradictions_data = [
            {
                "contradiction_type": contra.contradiction_type.value,
                "statement_a": contra.statement_a,
                "statement_b": contra.statement_b,
                "confidence": contra.confidence,
                "explanation": contra.explanation,
            }
            for contra in contradictions
        ]

        response_data = {
            "success": True,
            "terminology": terminology_data,
            "contradictions": contradictions_data,
        }

        serialized = convert_numpy_types(response_data)
        return jsonify(serialized)

    except Exception as e:
        import traceback

        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error("analyze-document error:\n%s", error_details)
        return jsonify({"error": str(e), "debug_details": error_details}), 500


@app.route("/export/json", methods=["POST"])
def export_json():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    import json as _json
    from flask import Response
    output = _json.dumps(data, indent=2, default=str)
    filename = (
        data.get("preprocessing", {}).get("metadata", {}).get("original_filename", "analysis")
    )
    base = re.sub(r"\.[^/.]+$", "", filename)
    return Response(
        output,
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{base}_report.json"'},
    )


@app.route("/export/csv", methods=["POST"])
def export_csv():
    import csv
    import io as _io
    from flask import Response

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    preprocessing = data.get("preprocessing", {})
    meta = preprocessing.get("metadata", {})
    proc_stats = data.get("processing", {}).get("processing_stats", {})
    semantic = data.get("semantic", {})
    analysis = data.get("analysis", {})
    term_stats = analysis.get("terminology", {}).get("statistics", {})
    contradictions = analysis.get("contradictions", [])

    buf = _io.StringIO()
    w = csv.writer(buf)

    w.writerow(["DOCUMENT STATISTICS"])
    w.writerow(["Filename", meta.get("original_filename", "Unknown")])
    w.writerow(["Text Length (chars)", meta.get("text_length", 0)])
    w.writerow(["Analyzed At", meta.get("processed_at", "Unknown")])
    w.writerow(["Total Segments", proc_stats.get("total_segments", 0)])
    w.writerow(["  Normative", proc_stats.get("normative_segments", 0)])
    w.writerow(["  Informative", proc_stats.get("informative_segments", 0)])
    w.writerow(["  Unknown", proc_stats.get("unknown_segments", 0)])
    w.writerow(["High Confidence Segments (≥0.7)", proc_stats.get("high_confidence_segments", 0)])
    w.writerow(["Medium Confidence Segments", proc_stats.get("medium_confidence_segments", 0)])
    w.writerow(["Low Confidence Segments (<0.4)", proc_stats.get("low_confidence_segments", 0)])
    w.writerow([])

    w.writerow(["SEMANTIC EXTRACTION"])
    w.writerow(["Total Profiles", semantic.get("total_profiles", 0)])
    avg_conf = semantic.get("avg_confidence") or 0
    w.writerow(["Average Profile Confidence", f"{avg_conf * 100:.1f}%"])
    w.writerow([])

    w.writerow(["TERMINOLOGY"])
    w.writerow(["Defined Terms", term_stats.get("defined_terms", 0)])
    w.writerow(["Undefined Terms", term_stats.get("undefined_terms", 0)])
    cov = term_stats.get("definition_coverage") or 0
    w.writerow(["Definition Coverage", f"{cov * 100:.1f}%"])
    w.writerow([])

    w.writerow(["CONTRADICTIONS"])
    w.writerow(["Total Detected", len(contradictions)])
    w.writerow([])
    if contradictions:
        w.writerow(["#", "Type", "Confidence", "Statement A", "Statement B"])
        for i, c in enumerate(contradictions, 1):
            w.writerow([
                i,
                c.get("contradiction_type", ""),
                f"{(c.get('confidence') or 0) * 100:.1f}%",
                c.get("statement_a", ""),
                c.get("statement_b", ""),
            ])

    filename = meta.get("original_filename", "analysis")
    base = re.sub(r"\.[^/.]+$", "", filename)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{base}_statistics.csv"'},
    )


@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
        from reportlab.lib.enums import TA_LEFT
    except ImportError:
        return jsonify({"error": "reportlab is not installed"}), 500

    import io as _io
    import html as _html

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    preprocessing = data.get("preprocessing", {})
    meta = preprocessing.get("metadata", {})
    proc_stats = data.get("processing", {}).get("processing_stats", {})
    semantic = data.get("semantic", {})
    profiles = semantic.get("profiles", [])
    analysis = data.get("analysis", {})
    term_stats = analysis.get("terminology", {}).get("statistics", {})
    contradictions = analysis.get("contradictions", [])

    ORANGE = colors.HexColor("#E87722")

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ClarusTitle", parent=styles["Title"], fontSize=20, spaceAfter=4)
    h1 = ParagraphStyle("ClarusH1", parent=styles["Heading1"], fontSize=13, textColor=ORANGE, spaceBefore=14, spaceAfter=4)
    body = ParagraphStyle("ClarusBody", parent=styles["Normal"], fontSize=9, spaceAfter=2)
    small = ParagraphStyle("ClarusSmall", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#666666"))

    def esc(s):
        return _html.escape(str(s or ""))

    def stats_table(rows):
        t = Table(rows, colWidths=[3 * inch, 2.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ORANGE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    story = []

    story.append(Paragraph("CLARUS Analysis Report", title_style))
    story.append(Paragraph(f"<b>Document:</b> {esc(meta.get('original_filename', 'Unknown'))}", body))
    story.append(Paragraph(f"<b>Analyzed:</b> {esc(meta.get('processed_at', 'Unknown'))}", body))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE))
    story.append(Spacer(1, 0.1 * inch))

    # --- 1. Document Processing ---
    story.append(Paragraph("1. Document Processing", h1))
    story.append(stats_table([
        ["Metric", "Value"],
        ["Total Segments", proc_stats.get("total_segments", 0)],
        ["Normative Segments", proc_stats.get("normative_segments", 0)],
        ["Informative Segments", proc_stats.get("informative_segments", 0)],
        ["Unknown Segments", proc_stats.get("unknown_segments", 0)],
        ["High Confidence (≥0.7)", proc_stats.get("high_confidence_segments", 0)],
        ["Low Confidence (<0.4)", proc_stats.get("low_confidence_segments", 0)],
        ["Text Length (chars)", meta.get("text_length", 0)],
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # --- 2. Semantic Extraction ---
    story.append(Paragraph("2. Semantic Extraction", h1))
    avg_conf = (semantic.get("avg_confidence") or 0) * 100
    story.append(stats_table([
        ["Metric", "Value"],
        ["Total Profiles", semantic.get("total_profiles", 0)],
        ["Average Confidence", f"{avg_conf:.1f}%"],
    ]))
    story.append(Spacer(1, 0.08 * inch))

    if profiles:
        story.append(Paragraph(f"Profiles ({len(profiles)})", ParagraphStyle("ph2", parent=h1, fontSize=10, spaceBefore=6)))
        prof_rows = [["#", "Sentence", "Subject", "Object", "Modality", "Condition"]]
        for i, p in enumerate(profiles, 1):
            sent = esc(p.get("original_sentence") or "")
            if len(sent) > 70:
                sent = sent[:67] + "…"
            prof_rows.append([
                str(i), sent,
                esc(p.get("subject") or "—"),
                esc(p.get("object") or "—"),
                esc(p.get("modality") or "—"),
                esc(p.get("condition") or "—"),
            ])
        pt = Table(prof_rows, colWidths=[0.25*inch, 2.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.05*inch])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ORANGE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(pt)
    story.append(Spacer(1, 0.15 * inch))

    # --- 3. Terminology ---
    story.append(Paragraph("3. Terminology Analysis", h1))
    cov = (term_stats.get("definition_coverage") or 0) * 100
    story.append(stats_table([
        ["Metric", "Value"],
        ["Defined Terms", term_stats.get("defined_terms", 0)],
        ["Undefined Terms", term_stats.get("undefined_terms", 0)],
        ["Definition Coverage", f"{cov:.1f}%"],
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # --- 4. Contradictions ---
    story.append(Paragraph("4. Contradiction Analysis", h1))
    story.append(Paragraph(f"Total contradictions detected: <b>{len(contradictions)}</b>", body))
    story.append(Spacer(1, 0.06 * inch))

    if contradictions:
        cont_rows = [["#", "Type", "Conf.", "Statement A", "Statement B"]]
        for i, c in enumerate(contradictions, 1):
            sa = esc(c.get("statement_a") or "")
            sb = esc(c.get("statement_b") or "")
            if len(sa) > 55: sa = sa[:52] + "…"
            if len(sb) > 55: sb = sb[:52] + "…"
            conf = (c.get("confidence") or 0) * 100
            cont_rows.append([str(i), esc(c.get("contradiction_type") or ""), f"{conf:.0f}%", sa, sb])
        ct = Table(cont_rows, colWidths=[0.25*inch, 0.9*inch, 0.45*inch, 2.45*inch, 2.45*inch])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ORANGE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ct)

    doc.build(story)
    buf.seek(0)

    filename = meta.get("original_filename", "analysis")
    base = re.sub(r"\.[^/.]+$", "", filename)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{base}_summary.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
