import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from clarus.analysis.document_processor import DocumentProcessor, ElementType


class TestNLPClassification:

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    def test_should_classify_strong_normative_requirements_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            ("The system shall process user data", ElementType.NORMATIVE),
            ("All users must authenticate", ElementType.NORMATIVE),
            ("The application will validate inputs", ElementType.NORMATIVE),
            ("Components shall be tested", ElementType.NORMATIVE),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type} using NLP"
            assert (
                confidence >= 0.85
            ), f"Should give high confidence for strong normative: {confidence}"

    def test_should_classify_weak_modalities_as_informative_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            ("The system should process efficiently", ElementType.INFORMATIVE),
            ("Users may access during business hours", ElementType.INFORMATIVE),
            ("The application can handle multiple formats", ElementType.INFORMATIVE),
            ("Components could be optimized", ElementType.INFORMATIVE),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type} using NLP"
            assert (
                confidence >= 0.60
            ), f"Should give medium confidence for weak modalities: {confidence}"

    def test_should_handle_negated_normative_correctly_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            ("The system shall not store personal data", ElementType.NORMATIVE),
            ("Users must not share passwords", ElementType.NORMATIVE),
            ("Applications shall not crash", ElementType.NORMATIVE),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify negated '{text}' as {expected_type} using NLP"
            assert (
                confidence <= 0.50
            ), f"Should give low confidence for negated normative: {confidence}"

    def test_should_classify_explicit_informative_content_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            ("For example, the system processes data", ElementType.INFORMATIVE),
            ("Note: Authentication is required", ElementType.INFORMATIVE),
            ("See section 3.1 for details", ElementType.INFORMATIVE),
            ("This illustrates the workflow", ElementType.INFORMATIVE),
            ("Remark: All fields are mandatory", ElementType.INFORMATIVE),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type} using NLP"
            assert (
                confidence >= 0.80
            ), f"Should give high confidence for explicit informative: {confidence}"

    def test_should_classify_questions_as_informative_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            ("What is the purpose of this module?", ElementType.INFORMATIVE),
            ("How does the system handle errors?", ElementType.INFORMATIVE),
            ("Where can users find documentation?", ElementType.INFORMATIVE),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify question '{text}' as {expected_type} using NLP"

    def test_should_handle_mixed_modalities_appropriately_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        text = "The system shall process data and should be efficient"
        result = processor._classify_line(text)
        confidence = processor._calculate_confidence(text, result)

        assert (
            result == ElementType.NORMATIVE
        ), f"Mixed modalities should be normative using NLP"
        assert (
            confidence >= 0.75
        ), f"Should give good confidence for mixed modalities: {confidence}"

    def test_should_classify_unknown_content_appropriately_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            ("This is some random text", ElementType.UNKNOWN),
            ("The system does something", ElementType.UNKNOWN),
            ("Users interact with interface", ElementType.UNKNOWN),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type} using NLP"
            assert (
                confidence <= 0.60
            ), f"Should give low confidence for unknown content: {confidence}"

    def test_should_fallback_to_regex_when_spacy_unavailable(self, processor):
        original_nlp = processor.nlp
        processor.nlp = None

        try:
            text = "The system shall process data"
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == ElementType.NORMATIVE
            ), f"Regex fallback should classify as normative"
            assert confidence > 0.0, f"Regex fallback should provide confidence"

            text = "For example, the system processes data"
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == ElementType.INFORMATIVE
            ), f"Regex fallback should classify as informative"
            assert confidence > 0.0, f"Regex fallback should provide confidence"

        finally:
            processor.nlp = original_nlp

    def test_should_extract_semantic_anchors_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        text = "The system shall immediately process user data when authentication is provided"
        anchors = processor.extract_semantic_anchors(text)

        assert anchors.modality == "shall", f"Should extract modality using NLP"
        assert anchors.temporal == "immediately", f"Should extract temporal using NLP"
        assert anchors.condition is not None, f"Should extract condition using NLP"
        assert "when" in anchors.condition, f"Condition should contain 'when' using NLP"
        assert anchors.confidence > 0.5, f"Should have reasonable confidence using NLP"

    def test_should_handle_complex_sentences_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        test_cases = [
            (
                "If the user is authenticated, the system shall process data within 5 seconds",
                ElementType.NORMATIVE,
            ),
            (
                "For example, when the system loads, it should display a welcome message",
                ElementType.INFORMATIVE,
            ),
            ("The system must not fail under normal conditions", ElementType.NORMATIVE),
        ]

        for text, expected_type in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify complex sentence using NLP: '{text}'"
            assert (
                confidence > 0.0
            ), f"Should provide confidence for complex sentence: {confidence}"

    def test_should_process_full_document_with_nlp(self, processor):
        if not processor.nlp:
            pytest.skip("SpaCy model not available - skipping NLP tests")
            return

        document_text = """
        1. System Requirements

        The system shall process user data within 5 seconds.
        All users must authenticate before access.

        2. Notes

        For example, the system uses encryption.
        Note: Passwords should be complex.
        """

        result = processor.process_document(document_text)

        assert (
            len(result.segments) > 0
        ), f"Should create segments using NLP, got {len(result.segments)}"

        normative_count = sum(
            1 for s in result.segments if s.element_type == ElementType.NORMATIVE
        )
        informative_count = sum(
            1 for s in result.segments if s.element_type == ElementType.INFORMATIVE
        )

        assert (
            normative_count >= 1
        ), f"Should find normative segments using NLP, found {normative_count}"
        assert (
            informative_count >= 2
        ), f"Should find informative segments using NLP, found {informative_count}"

        normative_segments = [
            s for s in result.segments if s.element_type == ElementType.NORMATIVE
        ]
        assert all(
            s.semantic_anchors is not None for s in normative_segments
        ), "Normative segments should have semantic anchors using NLP"

        assert result.processing_stats["total_segments"] == len(result.segments)
        assert result.processing_stats["normative_segments"] == normative_count
        assert result.processing_stats["informative_segments"] == informative_count
