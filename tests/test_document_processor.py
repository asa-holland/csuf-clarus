import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from clarus.analysis.document_processor import (
    DocumentProcessor,
    ElementType,
    DocumentSegment,
    ProcessedDocument,
    SemanticAnchor,
)


class TestDocumentProcessor:

    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_should_initialize_processor_with_required_components(self):
        processor = DocumentProcessor()
        assert processor.normative_regex is not None
        assert processor.informative_regex is not None
        assert len(processor.heading_patterns) == 3

    def test_should_classify_normative_lines_correctly(self):
        normative_lines = [
            "The system shall process the data.",
            "Users must authenticate before access.",
            "The operator shall not exceed limits.",
            "All components is required to pass inspection.",
        ]

        for line in normative_lines:
            element_type = self.processor._classify_line(line)
            assert element_type == ElementType.NORMATIVE

    def test_should_classify_informative_lines_correctly(self):
        informative_lines = [
            "Users should verify credentials.",
            "The system may log activities.",
            "For example, consider the following case.",
            "Note: This is an important consideration.",
            "This component can handle multiple inputs.",
        ]

        for line in informative_lines:
            element_type = self.processor._classify_line(line)
            assert element_type == ElementType.INFORMATIVE

    def test_should_classify_unknown_lines_correctly(self):
        unknown_lines = [
            "This is a regular sentence.",
            "The component processes information.",
            "System architecture overview.",
        ]

        for line in unknown_lines:
            element_type = self.processor._classify_line(line)
            assert element_type == ElementType.UNKNOWN

    def test_should_identify_numbered_headings(self):
        heading_line = "1.2 System Requirements"
        result = self.processor._identify_heading(heading_line)

        assert result is not None
        assert result[0] == "1.2"
        assert result[1] == "System Requirements"

    def test_should_identify_all_caps_headings(self):
        heading_line = "INTRODUCTION"
        result = self.processor._identify_heading(heading_line)

        assert result is not None
        assert result[0] is None
        assert result[1] == "INTRODUCTION"

    def test_should_not_identify_regular_lines_as_headings(self):
        regular_line = "This is a regular sentence in the document."
        result = self.processor._identify_heading(regular_line)

        assert result is None

    def test_should_calculate_high_confidence_for_normative_lines(self):
        line = "The system shall process the data."
        confidence = self.processor._calculate_confidence(line, ElementType.NORMATIVE)

        assert confidence >= 0.7

    def test_should_calculate_high_confidence_for_informative_lines(self):
        line = "Users should verify their credentials."
        confidence = self.processor._calculate_confidence(line, ElementType.INFORMATIVE)

        assert confidence >= 0.6

    def test_should_calculate_low_confidence_for_unknown_lines(self):
        line = "This is a regular sentence."
        confidence = self.processor._calculate_confidence(line, ElementType.UNKNOWN)

        assert confidence < 0.5

    def test_should_process_simple_document_successfully(self):
        document_text = """1. Introduction
This document describes system requirements.

2. Requirements
The system shall process user data.
Users must authenticate before access.
Note: Authentication should be secure.

3. Examples
For example, consider the following case."""

        result = self.processor.process_document(document_text)

        assert isinstance(result, ProcessedDocument)
        assert result.original_text == document_text
        assert len(result.segments) > 0
        assert result.processing_stats["total_segments"] > 0

    def test_should_generate_processing_statistics(self):
        document_text = """1. Requirements
The system shall process data.
Users must authenticate.
Note: This is important.
Users should verify credentials."""

        result = self.processor.process_document(document_text)
        stats = result.processing_stats

        assert stats["total_segments"] > 0
        assert stats["normative_segments"] >= 2
        assert stats["informative_segments"] >= 2
        assert "high_confidence_segments" in stats
        assert "medium_confidence_segments" in stats
        assert "low_confidence_segments" in stats

    def test_should_extract_normative_segments(self):
        document_text = """The system shall process data.
Users should verify credentials.
Note: This is important.
All components must pass inspection."""

        processed = self.processor.process_document(document_text)
        normative_segments = self.processor.get_normative_segments(processed)

        assert len(normative_segments) >= 2
        for segment in normative_segments:
            assert segment.element_type == ElementType.NORMATIVE

    def test_should_extract_informative_segments(self):
        document_text = """The system shall process data.
Users should verify credentials.
Note: This is important.
For example, consider this case."""

        processed = self.processor.process_document(document_text)
        informative_segments = self.processor.get_informative_segments(processed)

        assert len(informative_segments) >= 2
        for segment in informative_segments:
            assert segment.element_type == ElementType.INFORMATIVE

    def test_should_handle_empty_document(self):
        result = self.processor.process_document("")

        assert isinstance(result, ProcessedDocument)
        assert result.original_text == ""
        assert len(result.segments) == 0
        assert result.processing_stats["total_segments"] == 0

    def test_should_process_document_with_metadata(self):
        metadata = {"title": "Test Document", "version": "1.0"}
        document_text = "The system shall process data."

        result = self.processor.process_document(document_text, metadata)

        assert result.metadata == metadata
        assert len(result.segments) > 0

    def test_should_extract_semantic_anchors(self):
        test_sentences = [
            "The system shall process the data.",
            "If the temperature exceeds 90°C, the operator must immediately initiate the emergency cooling sequence.",
            "Users must authenticate before access.",
            "The system shall not exceed limits.",
        ]

        for sentence in test_sentences:
            anchors = self.processor.extract_semantic_anchors(sentence)
            assert isinstance(anchors, SemanticAnchor)
            assert anchors.confidence >= 0.0
            assert anchors.confidence <= 1.0

    def test_should_detect_modality_in_semantic_anchors(self):
        modalities = {
            "The system shall process data.": "shall",
            "Users must authenticate.": "must",
            "The system should verify credentials.": "should",
            "Users may access the system.": "may",
        }

        for sentence, expected_modality in modalities.items():
            anchors = self.processor.extract_semantic_anchors(sentence)
            assert anchors.modality == expected_modality

    def test_should_detect_negation_in_semantic_anchors(self):
        negated_sentences = [
            "The system shall not exceed limits.",
            "Users must never share credentials.",
            "The system cannot process without authentication.",
        ]

        for sentence in negated_sentences:
            anchors = self.processor.extract_semantic_anchors(sentence)
            assert anchors.negation is not None

    def test_should_extract_conditions_from_semantic_anchors(self):
        conditional_sentences = [
            "If the temperature exceeds 90°C, the system shall activate cooling.",
            "When users authenticate, the system must log the activity.",
            "Unless authorized, access is prohibited.",
        ]

        for sentence in conditional_sentences:
            anchors = self.processor.extract_semantic_anchors(sentence)
            assert anchors.condition is not None
            assert any(
                indicator in anchors.condition.lower()
                for indicator in ["if", "when", "unless"]
            )

    def test_should_extract_temporal_information_from_semantic_anchors(self):
        temporal_sentences = [
            "The system must respond immediately.",
            "Users shall authenticate within 5 minutes.",
            "The system must reboot before midnight.",
        ]

        for sentence in temporal_sentences:
            anchors = self.processor.extract_semantic_anchors(sentence)
            assert anchors.temporal is not None

    def test_should_extract_subject_and_object_from_semantic_anchors(self):
        sentence = "The operator shall initiate the emergency sequence."
        anchors = self.processor.extract_semantic_anchors(sentence)

        assert anchors.subject is not None or anchors.object is not None

    def test_should_include_semantic_anchors_in_processed_document(self):
        document_text = """1. Requirements
The system shall process user data.
Users must authenticate before access.
Note: This is important."""

        result = self.processor.process_document(document_text)

        normative_segments = self.processor.get_normative_segments(result)
        for segment in normative_segments:
            assert segment.semantic_anchors is not None
            assert isinstance(segment.semantic_anchors, SemanticAnchor)

    def test_should_calculate_semantic_anchor_confidence(self):
        simple_sentence = "The system shall process data."
        anchors = self.processor.extract_semantic_anchors(simple_sentence)

        assert anchors.confidence > 0.0

        complex_sentence = "If the temperature exceeds 90°C, the operator must immediately initiate the emergency cooling sequence."
        complex_anchors = self.processor.extract_semantic_anchors(complex_sentence)

        assert complex_anchors.confidence >= anchors.confidence

    def test_should_handle_multiline_segments(self):
        document_text = """1. Requirements
The system shall process user data
and store it securely.
Users must authenticate before access
to ensure proper security."""

        result = self.processor.process_document(document_text)

        multiline_segments = [s for s in result.segments if "\n" in s.text]
        assert len(multiline_segments) > 0

        for segment in result.segments:
            assert segment.start_line <= segment.end_line
            assert segment.start_line > 0
