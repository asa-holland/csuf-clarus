"""
Unit tests for the modern_terminology_validator module.

Tests follow the format test_should_xyz and use pytest framework.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, List

from clarus.analysis.modern_terminology_validator import (
    ModernTerminologyValidator,
    TermClassification,
    DefinitionDetection,
    ModernValidationResult,
    ModernValidationReport,
)


class TestModernTerminologyValidator:
    """Test cases for ModernTerminologyValidator class"""

    @pytest.fixture
    def sample_glossary(self) -> Dict[str, str]:
        """Sample glossary for testing"""
        return {
            "API": "Application Programming Interface",
            "Authentication": "Process of verifying identity",
            "Database": "Organized collection of structured information",
        }

    @pytest.fixture
    def validator_without_transformers(self, sample_glossary):
        """Create validator instance without loading transformers"""
        with patch(
            "clarus.analysis.modern_terminology_validator.TRANSFORMERS_AVAILABLE",
            False,
        ):
            return ModernTerminologyValidator(glossary=sample_glossary)

    @pytest.fixture
    def validator_with_mock_transformers(self, sample_glossary):
        """Create validator with mocked transformer models"""
        with patch(
            "clarus.analysis.modern_terminology_validator.TRANSFORMERS_AVAILABLE",
            True,
        ):
            with patch(
                "clarus.analysis.modern_terminology_validator.torch.cuda.is_available",
                return_value=False,
            ):
                validator = ModernTerminologyValidator(glossary=sample_glossary)

                # Mock the transformer models
                validator.term_extraction_model = Mock()
                validator.definition_model = Mock()
                validator.semantic_model = Mock()

                return validator

    def test_should_initialize_with_empty_glossary(self):
        """Test that validator initializes correctly with empty glossary"""
        with patch(
            "clarus.analysis.modern_terminology_validator.TRANSFORMERS_AVAILABLE",
            False,
        ):
            validator = ModernTerminologyValidator()

            assert validator.glossary == {}
            assert validator._term_index == {}
            assert validator.device == "cpu"

    def test_should_build_term_index_correctly(
        self, validator_without_transformers, sample_glossary
    ):
        """Test that term index is built correctly from glossary"""
        validator = validator_without_transformers

        # Check that terms are indexed
        assert "api" in validator._term_index
        assert "authentication" in validator._term_index
        assert "database" in validator._term_index

        # Check that plural forms are indexed
        assert "apis" in validator._term_index
        assert "authentications" in validator._term_index
        assert "databases" in validator._term_index

    def test_should_classify_common_english_word_rule_based(
        self, validator_without_transformers
    ):
        """Test classification of common English words using rule-based fallback"""
        validator = validator_without_transformers
        result = validator._classify_term_rule_based("the", None)

        assert result.term == "the"
        assert result.is_common_english is True
        assert result.is_domain_term is False
        assert result.confidence == 0.6
        assert result.classification_reason == "rule_based_fallback"

    def test_should_classify_acronym_as_domain_term(
        self, validator_without_transformers
    ):
        """Test classification of acronyms as domain terms"""
        validator = validator_without_transformers
        result = validator._classify_term_rule_based("API", None)

        assert result.term == "API"
        assert result.is_common_english is False
        assert result.is_domain_term is True
        assert result.confidence == 0.8

    def test_should_classify_camelcase_as_domain_term(
        self, validator_without_transformers
    ):
        """Test classification of CamelCase terms as domain terms"""
        validator = validator_without_transformers
        result = validator._classify_term_rule_based("AuthenticationService", None)

        assert result.term == "AuthenticationService"
        assert result.is_common_english is False
        assert result.is_domain_term is True
        assert result.confidence == 0.8

    def test_should_classify_multi_word_term_as_domain_term(
        self, validator_without_transformers
    ):
        """Test classification of multi-word terms as domain terms"""
        validator = validator_without_transformers
        result = validator._classify_term_rule_based("user authentication", None)

        assert result.term == "user authentication"
        assert result.is_common_english is False
        assert result.is_domain_term is True
        assert result.confidence == 0.8

    def test_should_detect_definition_with_is_pattern(
        self, validator_without_transformers
    ):
        """Test definition detection using 'is' pattern"""
        validator = validator_without_transformers
        context = "Authentication is the process of verifying user identity"
        result = validator._detect_definition_rule_based("Authentication", context)

        assert result.term == "Authentication"
        assert result.has_definition is True
        assert "process of verifying user identity" in result.definition_text
        assert result.confidence == 0.8

    def test_should_detect_definition_with_means_pattern(
        self, validator_without_transformers
    ):
        """Test definition detection using 'means' pattern"""
        validator = validator_without_transformers
        context = "API means Application Programming Interface"
        result = validator._detect_definition_rule_based("API", context)

        assert result.term == "API"
        assert result.has_definition is True
        assert "Application Programming Interface" in result.definition_text

    def test_should_not_detect_definition_when_not_present(
        self, validator_without_transformers
    ):
        """Test that definition detection returns false when no definition is present"""
        validator = validator_without_transformers
        context = "The API handles requests from clients"
        result = validator._detect_definition_rule_based("API", context)

        assert result.term == "API"
        assert result.has_definition is False
        assert result.confidence == 0.3

    def test_should_validate_defined_term(self, validator_without_transformers):
        """Test validation of a term that exists in glossary"""
        validator = validator_without_transformers
        result = validator.validate_term_modern("API")

        assert result.term == "API"
        assert result.is_defined is True
        assert result.suggested_definition == "Application Programming Interface"

    def test_should_validate_undefined_domain_term(
        self, validator_without_transformers
    ):
        """Test validation of an undefined domain term"""
        validator = validator_without_transformers
        result = validator.validate_term_modern("AuthenticationService")

        assert result.term == "AuthenticationService"
        assert result.is_defined is False
        assert result.classification.is_domain_term is True
        assert result.classification.is_common_english is False

    def test_should_validate_common_english_term(self, validator_without_transformers):
        """Test validation of a common English term"""
        validator = validator_without_transformers
        result = validator.validate_term_modern("system")

        assert result.term == "system"
        assert result.is_defined is False
        assert result.classification.is_common_english is True
        assert result.classification.is_domain_term is False

    def test_should_validate_multiple_terms(self, validator_without_transformers):
        """Test validation of multiple terms"""
        validator = validator_without_transformers
        terms = ["API", "system", "Blockchain", "the"]
        report = validator.validate_terms_modern(terms)

        assert isinstance(report, ModernValidationReport)
        assert len(report.defined_terms) == 1  # API
        assert len(report.common_english_terms) == 2  # system, the
        assert len(report.undefined_terms) == 1  # Blockchain
        assert report.statistics["total_terms"] == 4
        assert report.statistics["defined_terms"] == 1

    def test_should_calculate_semantic_similarity_zero_when_no_model(
        self, validator_without_transformers
    ):
        """Test semantic similarity calculation when no model is available"""
        validator = validator_without_transformers
        similarity = validator._calculate_semantic_similarity("API", "interface")

        assert similarity == 0.0

    def test_should_extract_definition_from_context(
        self, validator_without_transformers
    ):
        """Test extraction of definition text from context"""
        validator = validator_without_transformers
        context = "Authentication is the process of verifying identity"
        definition = validator._extract_definition_from_context(
            "Authentication", context
        )

        assert definition == "the process of verifying identity"

    def test_should_return_none_when_no_definition_extracted(
        self, validator_without_transformers
    ):
        """Test that None is returned when no definition can be extracted"""
        validator = validator_without_transformers
        context = "The system processes authentication requests"
        definition = validator._extract_definition_from_context(
            "Authentication", context
        )

        assert definition is None

    def test_should_create_empty_semantic_clusters_when_no_model(
        self, validator_without_transformers
    ):
        """Test that empty semantic clusters are created when no model is available"""
        validator = validator_without_transformers
        undefined_terms = [
            ModernValidationResult("term1", "term1", False, Mock(), None, 0.0),
            ModernValidationResult("term2", "term2", False, Mock(), None, 0.0),
        ]

        clusters = validator._create_semantic_clusters(undefined_terms)

        assert clusters == {}

    def test_should_get_priority_undefined_terms_empty_list(
        self, validator_without_transformers
    ):
        """Test getting priority undefined terms when list is empty"""
        validator = validator_without_transformers
        report = ModernValidationReport([], [], [], [], {})

        priority_terms = validator.get_priority_undefined_terms(report)

        assert priority_terms == []

    def test_should_get_priority_undefined_terms_sorted_by_score(
        self, validator_without_transformers
    ):
        """Test that priority undefined terms are sorted by priority score"""
        validator = validator_without_transformers

        # Create mock results
        result1 = ModernValidationResult(
            "term1",
            "term1",
            False,
            Mock(is_domain_term=True, confidence=0.8),
            None,
            0.0,
        )
        result2 = ModernValidationResult(
            "term2",
            "term2",
            False,
            Mock(is_domain_term=True, confidence=0.9),
            None,
            0.0,
        )

        report = ModernValidationReport(
            [], [result1, result2], [], [result1, result2], {}
        )

        priority_terms = validator.get_priority_undefined_terms(report, top_k=2)

        assert len(priority_terms) == 2
        assert (
            priority_terms[0]["term"] == "term2"
        )  # Higher confidence should come first
        assert priority_terms[1]["term"] == "term1"

    def test_should_add_glossary_terms_and_rebuild_index(
        self, validator_without_transformers
    ):
        """Test adding new glossary terms and rebuilding the index"""
        validator = validator_without_transformers

        # Add new terms
        new_terms = {"Token": "Security token for authentication"}
        validator.add_glossary_terms(new_terms)

        # Check that terms were added
        assert "Token" in validator.glossary
        assert "token" in validator._term_index
        assert validator._term_index["token"] == "Token"

    def test_should_handle_empty_term_list(self, validator_without_transformers):
        """Test validation of empty term list"""
        validator = validator_without_transformers
        report = validator.validate_terms_modern([])

        assert isinstance(report, ModernValidationReport)
        assert report.statistics["total_terms"] == 0
        assert len(report.defined_terms) == 0
        assert len(report.undefined_terms) == 0


class TestModernTerminologyValidatorWithMocks:
    """Test cases that use mocked transformer models"""

    @pytest.fixture
    def mock_validator(self):
        """Create validator with mocked transformer models"""
        with patch(
            "clarus.analysis.modern_terminology_validator.TRANSFORMERS_AVAILABLE",
            True,
        ):
            with patch(
                "clarus.analysis.modern_terminology_validator.torch.cuda.is_available",
                return_value=False,
            ):
                validator = ModernTerminologyValidator()

                # Mock all transformer components
                validator.term_extraction_tokenizer = Mock()
                validator.term_extraction_model = Mock()
                validator.definition_tokenizer = Mock()
                validator.definition_model = Mock()
                validator.semantic_model = Mock()

                return validator

    def test_should_use_transformer_classification_when_available(self, mock_validator):
        """Test that transformer classification is used when models are available"""
        # Mock transformer outputs
        mock_inputs = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        mock_validator.term_extraction_tokenizer.return_value = mock_inputs

        mock_outputs = Mock()
        mock_logits = Mock()
        mock_logits.logits = Mock()
        mock_outputs.logits = mock_logits

        mock_validator.term_extraction_model.return_value = mock_outputs

        # Mock torch operations
        with patch(
            "clarus.analysis.modern_terminology_validator.torch.nn.functional.softmax"
        ) as mock_softmax:
            with patch(
                "clarus.analysis.modern_terminology_validator.torch.max"
            ) as mock_max:
                mock_softmax.return_value = Mock()
                mock_max.return_value = Mock()
                mock_max.return_value.item.return_value = 0.8

                result = mock_validator._classify_term_with_transformer("API", None)

                assert result.term == "API"
                assert result.confidence == 0.8
                mock_validator.term_extraction_tokenizer.assert_called_once()

    def test_should_fallback_to_rule_based_on_transformer_error(self, mock_validator):
        """Test fallback to rule-based classification when transformer fails"""
        # Mock transformer to raise an exception
        mock_validator.term_extraction_tokenizer.side_effect = Exception("Model error")

        # Mock the rule-based method
        with patch.object(
            mock_validator, "_classify_term_rule_based"
        ) as mock_rule_based:
            mock_rule_based.return_value = TermClassification(
                term="API",
                is_domain_term=True,
                is_common_english=False,
                confidence=0.8,
                classification_reason="rule_based_fallback",
            )

            result = mock_validator._classify_term_with_transformer("API", None)

            assert result.classification_reason == "rule_based_fallback"
            mock_rule_based.assert_called_once_with("API", None)

    def test_should_use_semantic_model_when_available(self, mock_validator):
        """Test that semantic model is used for embeddings when available"""
        mock_validator.semantic_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        embedding = mock_validator._get_term_embedding("API")

        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        mock_validator.semantic_model.encode.assert_called_once_with(
            "API", convert_to_numpy=True
        )

    def test_should_handle_semantic_model_error(self, mock_validator):
        """Test handling of semantic model errors"""
        mock_validator.semantic_model.encode.side_effect = Exception("Model error")

        embedding = mock_validator._get_term_embedding("API")

        assert embedding is None

    def test_should_calculate_semantic_similarity_with_model(self, mock_validator):
        """Test semantic similarity calculation when model is available"""
        mock_validator.semantic_model.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.2, 0.3, 0.4]),
        ]

        similarity = mock_validator._calculate_semantic_similarity("API", "interface")

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_should_create_semantic_clusters_with_model(self, mock_validator):
        """Test semantic clustering when model is available"""
        # Create mock results
        result1 = ModernValidationResult("term1", "term1", False, Mock(), None, 0.0)
        result2 = ModernValidationResult("term2", "term2", False, Mock(), None, 0.0)

        # Mock semantic model to return similar embeddings
        mock_validator.semantic_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        clusters = mock_validator._create_semantic_clusters([result1, result2])

        assert isinstance(clusters, dict)
        # Should create at least one cluster when embeddings are identical
