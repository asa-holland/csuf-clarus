import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from clarus.analysis.document_processor import DocumentProcessor, ElementType


class TestSpacyIntegration:

    def test_should_load_spacy_model_successfully(self):
        processor = DocumentProcessor()

        assert (
            processor.nlp is not None
        ), "SpaCy model should be loaded in Docker environment"

        doc = processor.nlp("The system shall process user data.")

        assert len(doc) > 0, "SpaCy should tokenize the text"
        assert any(
            token.text == "shall" for token in doc
        ), "SpaCy should find 'shall' token"

        tokens = [
            token for token in doc if token.lemma_.lower() in processor.modality_verbs
        ]
        assert len(tokens) > 0, "SpaCy should find modality verbs"

    def test_should_use_nlp_for_classification(self):
        processor = DocumentProcessor()

        if processor.nlp is None:
            pytest.skip("SpaCy not available - this test requires NLP")
            return

        text = "The system shall process user data securely."
        result = processor._classify_line(text)
        confidence = processor._calculate_confidence(text, result)

        assert result == ElementType.NORMATIVE, "NLP should classify as normative"
        assert (
            confidence >= 0.85
        ), "NLP should give high confidence for strong normative"

        text = "For example, the system processes data in real-time."
        result = processor._classify_line(text)
        confidence = processor._calculate_confidence(text, result)

        assert result == ElementType.INFORMATIVE, "NLP should classify as informative"
        assert (
            confidence >= 0.80
        ), "NLP should give high confidence for explicit informative"

    def test_should_extract_semantic_anchors_with_nlp(self):
        processor = DocumentProcessor()

        if processor.nlp is None:
            pytest.skip("SpaCy not available - this test requires NLP")
            return

        text = "The system shall immediately process data when user is authenticated"
        anchors = processor.extract_semantic_anchors(text)

        assert anchors.modality == "shall", "Should extract modality"
        assert anchors.temporal == "immediately", "Should extract temporal"
        assert anchors.condition is not None, "Should extract condition"
        assert "when" in anchors.condition, "Condition should contain 'when'"
        assert anchors.confidence > 0.5, "Should have reasonable confidence"

    def test_should_handle_negation_with_nlp(self):
        processor = DocumentProcessor()

        if processor.nlp is None:
            pytest.skip("SpaCy not available - this test requires NLP")
            return

        text = "The system shall not store personal information."
        result = processor._classify_line(text)
        confidence = processor._calculate_confidence(text, result)

        assert result == ElementType.NORMATIVE, "Should still classify as normative"
        assert confidence <= 0.50, "Should give low confidence for negated normative"

    def test_should_process_full_document_with_nlp(self):
        processor = DocumentProcessor()

        if processor.nlp is None:
            pytest.skip("SpaCy not available - this test requires NLP")
            return

        document_text = """
        1. Requirements

        The system shall process user data within 5 seconds.
        All users must authenticate before access.

        2. Notes

        For example, the system uses encryption.
        Note: Passwords should be complex.
        """

        result = processor.process_document(document_text)

        assert len(result.segments) >= 4, "Should create multiple segments"

        normative_count = sum(
            1 for s in result.segments if s.element_type == ElementType.NORMATIVE
        )
        informative_count = sum(
            1 for s in result.segments if s.element_type == ElementType.INFORMATIVE
        )

        assert (
            normative_count >= 2
        ), f"Should find normative segments, found {normative_count}"
        assert (
            informative_count >= 2
        ), f"Should find informative segments, found {informative_count}"

        normative_segments = [
            s for s in result.segments if s.element_type == ElementType.NORMATIVE
        ]
        assert all(
            s.semantic_anchors is not None for s in normative_segments
        ), "Normative segments should have semantic anchors"

        assert result.processing_stats["total_segments"] == len(result.segments)
        assert result.processing_stats["normative_segments"] == normative_count
        assert result.processing_stats["informative_segments"] == informative_count
