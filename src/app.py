from flask import Flask, request, render_template, jsonify, send_file
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime
from clarus.steps.preprocess import extract_text_from_file, allowed_file
from clarus.steps.corpus_operations import create_corpus_zip, preprocess_corpus_files
from clarus.analysis.document_processor import DocumentProcessor
from clarus.analysis.semantic_extractor import SemanticExtractor
from clarus.analysis.modern_terminology_validator import ModernTerminologyValidator
from clarus.analysis.contradiction_analyzer import ContradictionDetector
import numpy as np


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
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
def index():
    return render_template("index.html")


@app.route("/pipeline")
def pipeline():
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

        modern_validator = ModernTerminologyValidator()
        contradiction_detector = ContradictionDetector(config={"enable_ml": False})

        terminology_result = modern_validator.validate_terms_modern([data["text"]])

        terminology_data = {
            "statistics": convert_numpy_types(terminology_result.statistics),
            "defined_terms": [
                {
                    "term": result.term,
                    "normalized_term": result.normalized_term,
                    "is_defined": result.is_defined,
                    "classification": {
                        "is_domain_term": result.classification.is_domain_term,
                        "is_common_english": result.classification.is_common_english,
                        "confidence": float(result.classification.confidence),
                    },
                    "suggested_definition": result.suggested_definition,
                }
                for result in terminology_result.defined_terms
            ],
            "undefined_terms": [
                {
                    "term": result.term,
                    "normalized_term": result.normalized_term,
                    "is_defined": result.is_defined,
                    "classification": {
                        "is_domain_term": result.classification.is_domain_term,
                        "is_common_english": result.classification.is_common_english,
                        "confidence": float(result.classification.confidence),
                    },
                    "suggested_definition": result.suggested_definition,
                }
                for result in terminology_result.undefined_terms
            ],
        }

        normative_statements = []
        if "segments" in data:
            for segment in data["segments"]:
                if segment.get("element_type") == "normative":
                    normative_statements.append(segment["text"])
        else:
            processor = DocumentProcessor()
            processed = processor.process_document(data["text"])
            normative_segments = processor.get_normative_segments(processed)
            normative_statements = [seg.text for seg in normative_segments]

        contradictions = contradiction_detector.detect_contradictions(
            normative_statements
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

        return jsonify(convert_numpy_types(response_data))

    except Exception as e:
        import traceback

        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG: analyze-document error:\n{error_details}")
        return jsonify({"error": str(e), "debug_details": error_details}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
