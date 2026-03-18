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
            if segment_data.get("element_type") == "normative":
                profile = extractor.extract_semantic_profile(segment_data["text"])
                profiles.append(
                    {
                        "original_sentence": profile.original_sentence,
                        "confidence": profile.confidence,
                        "modality": profile.modality.text if profile.modality else None,
                        "condition": (
                            profile.condition.text if profile.condition else None
                        ),
                        "subject": profile.subject.text if profile.subject else None,
                        "object": profile.object.text if profile.object else None,
                        "temporal": profile.temporal.text if profile.temporal else None,
                        "negation": profile.negation.text if profile.negation else None,
                    }
                )
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
            high_quality_terms = extracted_terms[:max_terms]

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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
