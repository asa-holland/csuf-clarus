# Clarus Documentation

A requirements document quality analysis tool developed in association with California State University, Fullerton's Master of Software Engineering program.

Clarus analyzes technical requirement documents to surface quality issues including undefined terminology, logical contradictions, modal inconsistencies, vague qualifiers, ambiguous references, and hidden normative statements: producing structured reports suitable for review or export.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Analysis Pipeline](#analysis-pipeline)
- [Configuration](#configuration)
- [Export Formats](#export-formats)
- [Development](#development)
- [Testing](#testing)
- [ML Models](#ml-models)

---

## Features

| Capability                  | Description                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| **Document Ingestion**      | Extracts text from PDF, DOCX, HTML, and TXT files                                             |
| **Segment Classification**  | Classifies each passage as normative, informative, or unknown                                 |
| **Semantic Extraction**     | Extracts subject, object, modality, condition, temporal, and negation profiles per sentence   |
| **Terminology Validation**  | Identifies undefined or inconsistently used terms using ML-based definition detection         |
| **Contradiction Detection** | Detects modality, temporal, conditional, semantic, terminological, and logical contradictions |
| **S2 Taxonomy Mapping**     | Maps findings to 8 structured error types across 4 severity categories                        |
| **Export**                  | Exports results as JSON, CSV, or PDF                                                          |

---

## Architecture

```
csuf-clarus/
├── src/
│   ├── app.py                          # Flask web server and API routes
│   ├── requirements.txt                # Python dependencies
│   ├── download_models.py              # Pre-downloads ML models at build time
│   ├── Dockerfile
│   └── clarus/
│       ├── steps/
│       │   ├── preprocess.py           # Text extraction (PDF, DOCX, HTML, TXT)
│       │   └── corpus_operations.py   # Batch corpus processing and ZIP export
│       ├── analysis/
│       │   ├── interfaces.py           # Abstract base classes and data contracts
│       │   ├── document_processor.py  # Segment classification (normative/informative)
│       │   ├── semantic_extractor.py  # Semantic profile extraction per sentence
│       │   ├── s2_taxonomy.py          # S2 error type definitions and severity levels
│       │   ├── terminology_validator.py # Term extraction, definition detection, clustering
│       │   └── contradiction_analyzer.py # Contradiction detection via NLI models
│       ├── templates/
│       │   └── pipeline.html           # Main web UI
│       └── static/css/
├── tests/                              # Pytest test suite (12 modules)
├── corpus_files/                       # Sample documents for corpus analysis
├── docker-compose.yml
└── pytest.ini
```

### Component Responsibilities

**Preprocessing**: `clarus/steps/preprocess.py`
Extracts raw text and metadata from uploaded files. Uses `pdfminer.six` for PDFs with layout analysis, `python-docx` for DOCX, `readability-lxml` + BeautifulSoup for HTML, and `chardet` for encoding-aware TXT parsing.

**Document Processor**: `clarus/analysis/document_processor.py`
Classifies each segment as normative (contains obligation modalities like *shall*, *must*) or informative (examples, notes, explanations). Uses SpaCy for POS tagging and dependency parsing when available; falls back to regex patterns.

**Semantic Extractor**: `clarus/analysis/semantic_extractor.py`
Extracts a semantic profile from each sentence: subject, object, modality type (obligation/prohibition/permission/recommendation/capability), conditional clause, temporal clause, and negation. Returns a confidence score per sentence.

**Terminology Validator**: `clarus/analysis/terminology_validator.py`
Uses a RoBERTa-based model to distinguish domain-specific terms from common English vocabulary. Detects whether each domain term has an explicit definition in the document. Clusters semantically related terms via sentence-transformers embeddings. Produces a validation report with defined/undefined term lists and coverage metrics.

**Contradiction Analyzer**: `clarus/analysis/contradiction_analyzer.py`
Runs every pair of normative segments through an NLI cross-encoder (`nli-deberta-v3-base`) to score entailment vs. contradiction. Applies rule-based pre-filters (overlapping subject terms, conflicting modalities) to reduce candidate pairs before model inference.

**S2 Taxonomy**: `clarus/analysis/s2_taxonomy.py`
Defines 8 named error types with severity levels:

| Error Type                | Category    | Severity |
| ------------------------- | ----------- | -------- |
| `UNDEFINED_TERM`          | Terminology | HIGH     |
| `SYNONYM_INCONSISTENCY`   | Terminology | MEDIUM   |
| `DIRECT_CONTRADICTION`    | Logical     | HIGH     |
| `MODAL_INCONSISTENCY`     | Logical     | HIGH     |
| `MISSING_TEMPORAL_ANCHOR` | Logical     | MEDIUM   |
| `VAGUE_QUALIFIERS`        | Vagueness   | MEDIUM   |
| `AMBIGUOUS_REFERENT`      | Vagueness   | MEDIUM   |
| `HIDDEN_NORMATIVE`        | Context     | LOW      |

---

## Quick Start

### Requirements

- Docker
- Docker Compose

### Run with Docker Compose

```bash
git clone <repository-url>
cd csuf-clarus
docker-compose up --build
```

The application is available at `http://localhost:5000`.

ML models are downloaded during the Docker build step. The first build takes several minutes; subsequent builds use the layer cache.

### Stop the Application

```bash
docker-compose down
```

### Development Mode

The `docker-compose.yml` is configured for development:
- Flask debug mode and live reload enabled
- Source directories mounted as volumes (code changes take effect immediately without rebuild)
- `FLASK_ENV=development`

---

## API Reference

All endpoints accept and return JSON unless otherwise noted.

### Health Check

```
GET /health
```

Returns `200 OK` with `{"status": "healthy"}` when the server is running.

---

### Upload Document

```
POST /upload
Content-Type: multipart/form-data
```

**Form field:** `file`: a PDF, DOCX, HTML, or TXT file.

**Response:**

```json
{
  "success": true,
  "text": "<extracted plain text>",
  "metadata": {
    "filename": "example.pdf",
    "format": "pdf",
    "char_count": 14823,
    "encoding": "utf-8"
  }
}
```

---

### Process Document

Segments text and classifies each segment as normative or informative.

```
POST /process-document
Content-Type: application/json
```

**Request:**

```json
{
  "text": "<document text>"
}
```

**Response:**

```json
{
  "success": true,
  "segments": [
    {
      "text": "The system shall authenticate users before granting access.",
      "element_type": "normative",
      "confidence": 0.95,
      "section_number": "3.1",
      "heading": "Authentication",
      "semantic_anchors": {
        "subject": "system",
        "modality": "obligation",
        "object": "users"
      }
    }
  ],
  "processing_stats": {
    "total_segments": 42,
    "normative_count": 28,
    "informative_count": 12,
    "unknown_count": 2
  }
}
```

---

### Extract Semantics

Extracts a semantic profile for each segment.

```
POST /extract-semantics
Content-Type: application/json
```

**Request:**

```json
{
  "segments": [ { "text": "...", "element_type": "normative" } ]
}
```

**Response:**

```json
{
  "success": true,
  "profiles": [
    {
      "original_sentence": "The system shall authenticate users before granting access.",
      "subject": "system",
      "modality": "obligation",
      "object": "users",
      "condition": null,
      "temporal": "before granting access",
      "negation": false,
      "confidence": 0.91
    }
  ],
  "total_profiles": 28,
  "avg_confidence": 0.87
}
```

---

### Analyze Document

Runs the full terminology and contradiction analysis pipeline.

```
POST /analyze-document
Content-Type: application/json
```

**Request:**

```json
{
  "text": "<document text>",
  "config": {
    "enable_terminology": true,
    "enable_contradictions": true,
    "detection_thresholds": {
      "min_confidence": 0.7,
      "ml_threshold": 0.8,
      "use_candidate_filter": true,
      "max_segment_distance": 50,
      "min_term_overlap": 1
    }
  }
}
```

**Detection threshold parameters:**

| Parameter              | Default | Description                                                 |
| ---------------------- | ------- | ----------------------------------------------------------- |
| `min_confidence`       | `0.7`   | Minimum confidence to include a finding                     |
| `ml_threshold`         | `0.8`   | NLI model score required to flag a contradiction            |
| `use_candidate_filter` | `true`  | Pre-filter pairs by subject overlap before ML inference     |
| `max_segment_distance` | `50`    | Maximum segment index distance between a contradicting pair |
| `min_term_overlap`     | `1`     | Minimum shared terms required for candidate pairing         |

**Response:**

```json
{
  "success": true,
  "terminology": {
    "defined_terms": ["authentication", "user"],
    "undefined_terms": ["principal", "credential store"],
    "common_english_terms": ["access", "system"],
    "semantic_clusters": [
      { "representative": "authentication", "members": ["auth", "login"] }
    ],
    "statistics": {
      "total_terms": 38,
      "defined_count": 21,
      "undefined_count": 17,
      "coverage_pct": 55.3
    }
  },
  "contradictions": [
    {
      "contradiction_type": "modality",
      "statement_a": "The system shall log all failed login attempts.",
      "statement_b": "The system shall not retain failed login attempt records.",
      "confidence": 0.94,
      "explanation": "Conflicting obligation and prohibition on the same subject.",
      "evidence": { "modality_a": "obligation", "modality_b": "prohibition" }
    }
  ]
}
```

---

### Export Results

```
POST /export/json
POST /export/csv
POST /export/pdf
Content-Type: application/json
```

**Request body:** The full analysis result object from `/analyze-document`.

**Responses:**
- `/export/json`: `application/json` file download
- `/export/csv`: `text/csv` file download with statistics table and contradiction list
- `/export/pdf`: `application/pdf` ReportLab-generated report

---

### Corpus Endpoints

```
GET  /download-corpus          # Download original corpus files as ZIP
GET  /preprocess-corpus        # Retrieve preprocessed corpus status
POST /preprocess-corpus        # Trigger batch preprocessing of corpus_files/
GET  /download-preprocessed    # Download preprocessed corpus as ZIP
```

---

## Analysis Pipeline

The typical call sequence for analyzing a document:

```
1. POST /upload            → extract text from file
2. POST /process-document  → segment and classify content
3. POST /extract-semantics → build semantic profiles (optional, for UI display)
4. POST /analyze-document  → run terminology + contradiction analysis
5. POST /export/pdf        → download report
```

Steps 2–4 can also be driven from the web UI at `http://localhost:5000`.

---

## Configuration

### Docker Environment Variables

Defined in `docker-compose.yml`:

| Variable               | Value                       | Purpose                                                      |
| ---------------------- | --------------------------- | ------------------------------------------------------------ |
| `FLASK_ENV`            | `development`               | Enables Flask debug mode                                     |
| `FLASK_DEBUG`          | `1`                         | Activates live reload                                        |
| `PYTHONUNBUFFERED`     | `1`                         | Unbuffered stdout for container logs                         |
| `HF_HOME`              | `/root/.cache/huggingface`  | Hugging Face model cache                                     |
| `TRANSFORMERS_CACHE`   | `/root/.cache/transformers` | Transformers model cache                                     |
| `TRANSFORMERS_OFFLINE` | `1`                         | Prevents runtime model downloads (use pre-downloaded models) |

### Allowed Upload Extensions

Defined in `src/clarus/steps/preprocess.py`: `pdf`, `docx`, `html`, `htm`, `txt`

---

## Export Formats

**JSON**: Full analysis result as pretty-printed JSON. All NaN/Inf float values are serialized as `null` for compatibility.

**CSV**: Two sections:
1. Statistics table (term counts, coverage percentage)
2. Contradiction list with type, confidence, and both statements

**PDF**: Formatted report via ReportLab with:
- Summary section (document metadata, analysis date)
- Terminology findings (defined/undefined term tables)
- Contradiction findings (one entry per detected contradiction with evidence)

---

## Development

### Running Without Docker

```bash
cd src
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python download_models.py   # pre-fetch sentence-transformer and NLI models
flask run
```

### Project Layout Conventions

- **`clarus/steps/`**: stateless I/O operations (file reading, format conversion)
- **`clarus/analysis/`**: stateful analysis components; each is a class with lazy ML model initialization
- **`clarus/analysis/interfaces.py`**: all abstract base classes and shared dataclasses; import from here to avoid circular dependencies
- **`app.py`**: thin Flask layer; delegates all logic to `clarus/` modules

### Adding a New Analysis Step

1. Define the input/output dataclasses in `interfaces.py`
2. Implement the step as a class in `clarus/analysis/`
3. Add a route in `app.py` that instantiates the class and calls it
4. Register the step in the UI pipeline in `templates/pipeline.html`

---

## Testing

```bash
# Inside the Docker container
docker-compose exec document-extractor pytest

# Run only fast tests
docker-compose exec document-extractor pytest -m "not slow and not integration"

# Run a specific module
docker-compose exec document-extractor pytest tests/test_contradiction_analyzer.py -v
```

### Test Markers

| Marker        | Description                                          |
| ------------- | ---------------------------------------------------- |
| `slow`        | Tests that load ML models or process large documents |
| `integration` | Tests that require a running Flask server            |

### Test Modules

| File                              | Coverage Area                                    |
| --------------------------------- | ------------------------------------------------ |
| `test_preprocess.py`              | Text extraction for all supported file formats   |
| `test_document_processor.py`      | Segment classification (normative/informative)   |
| `test_classification_nlp.py`      | NLP-driven element classification                |
| `test_semantic_extractor.py`      | Semantic profile extraction                      |
| `test_terminology_validator.py`   | Term detection and definition coverage           |
| `test_corpus_operations.py`       | Corpus batch processing and ZIP generation       |
| `test_nlp_verification.py`        | SpaCy model availability and pipeline validation |
| `test_spacy_integration_nlp.py`   | SpaCy integration with document processor        |
| `test_document_segment_errors.py` | Error detection in segments                      |
| `test_ui_consistency.py`          | Frontend template consistency                    |
| `test_ui_data_display.py`         | Data rendering in the web UI                     |

---

## ML Models

Clarus uses three pre-downloaded models. They are fetched during `docker-compose up --build` by `src/download_models.py` and stored in the container image layer.

| Model                 | Source                       | Used For                                               |
| --------------------- | ---------------------------- | ------------------------------------------------------ |
| `en_core_web_sm`      | SpaCy                        | Tokenization, POS tagging, dependency parsing          |
| `all-MiniLM-L6-v2`    | sentence-transformers        | Semantic embeddings for term clustering and similarity |
| `nli-deberta-v3-base` | cross-encoder (Hugging Face) | Natural language inference for contradiction detection |

NLTK data pre-downloaded: `punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`.

If models are unavailable at runtime, the document processor degrades gracefully to regex-based classification, and contradiction detection is skipped. Terminology validation requires the sentence-transformers model and will raise an error if it is absent.

---

## License

Apache 2.0: see [LICENSE](LICENSE).
