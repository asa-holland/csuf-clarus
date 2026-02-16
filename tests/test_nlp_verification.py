import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from clarus.analysis.document_processor import DocumentProcessor, ElementType


class TestNLPVerification:

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    def test_should_verify_spacy_model_availability(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - cannot verify NLP functionality")
            return

        test_doc = processor.nlp("The system shall process data.")
        assert len(test_doc) > 0, "SpaCy should tokenize text"
        assert hasattr(test_doc[0], "pos_"), "SpaCy should provide POS tags"
        assert hasattr(test_doc[0], "dep_"), "SpaCy should provide dependency parsing"

    def test_should_classify_strong_normative_examples(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - skipping NLP classification tests")
            return

        test_cases = [
            ("The system shall process user data", ElementType.NORMATIVE, "High"),
            ("All users must authenticate", ElementType.NORMATIVE, "High"),
            ("The application will validate inputs", ElementType.NORMATIVE, "High"),
        ]

        for text, expected_type, expected_confidence in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type}"
            assert (
                confidence >= 0.85
            ), f"Should give high confidence for strong normative"

    def test_should_classify_informative_examples(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - skipping NLP classification tests")
            return

        test_cases = [
            ("For example, the system processes data", ElementType.INFORMATIVE, "High"),
            ("Note: Authentication is required", ElementType.INFORMATIVE, "High"),
            (
                "Users may access during business hours",
                ElementType.INFORMATIVE,
                "Medium",
            ),
            ("What is the purpose?", ElementType.INFORMATIVE, "High"),
        ]

        for text, expected_type, expected_confidence in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type}"
            if expected_confidence == "High":
                assert (
                    confidence >= 0.80
                ), f"Should give high confidence for explicit informative"
            elif expected_confidence == "Medium":
                assert (
                    confidence >= 0.60
                ), f"Should give medium confidence for weak modalities"

    def test_should_classify_negated_normative_examples(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - skipping NLP classification tests")
            return

        test_cases = [
            ("The system shall not store passwords", ElementType.NORMATIVE, "Low"),
            ("Users must not share credentials", ElementType.NORMATIVE, "Low"),
        ]

        for text, expected_type, expected_confidence in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify negated '{text}' as {expected_type}"
            assert (
                confidence <= 0.50
            ), f"Should give low confidence for negated normative"

    def test_should_classify_unknown_examples(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - skipping NLP classification tests")
            return

        test_cases = [
            ("This is some random text", ElementType.UNKNOWN, "Low"),
            ("The system does something", ElementType.UNKNOWN, "Low"),
        ]

        for text, expected_type, expected_confidence in test_cases:
            result = processor._classify_line(text)
            confidence = processor._calculate_confidence(text, result)

            assert (
                result == expected_type
            ), f"Should classify '{text}' as {expected_type}"
            assert confidence <= 0.60, f"Should give low confidence for unknown content"

    def test_should_extract_semantic_anchors_correctly(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - skipping semantic anchor tests")
            return

        text = "The system shall immediately process user data when authentication is provided"
        anchors = processor.extract_semantic_anchors(text)

        assert anchors.modality == "shall", "Should extract modality"
        assert anchors.temporal == "immediately", "Should extract temporal"
        assert anchors.condition is not None, "Should extract condition"
        assert "when" in anchors.condition, "Should extract condition with 'when'"
        assert anchors.confidence > 0.5, "Should provide reasonable confidence"

    def test_should_handle_mixed_modalities(self, processor):
        if processor.nlp is None:
            pytest.skip("SpaCy model not available - skipping NLP classification tests")
            return

        text = "The system shall process data and should be efficient"
        result = processor._classify_line(text)
        confidence = processor._calculate_confidence(text, result)

        assert result == ElementType.NORMATIVE, "Mixed modalities should be normative"
        assert confidence >= 0.75, "Should give good confidence for mixed modalities"
