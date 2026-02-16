import pytest
from clarus.analysis.modern_terminology_validator import ModernTerminologyValidator


class TestModernTerminologyValidator:

    @pytest.fixture
    def sample_glossary(self):
        return {
            "authentication": "The process of verifying identity.",
            "API": "Application Programming Interface",
            "Database": "Organized collection of structured information",
        }

    @pytest.fixture
    def analyzer(self, sample_glossary):
        return ModernTerminologyValidator(glossary=sample_glossary)

    def test_should_initialize_with_empty_glossary(self):
        validator = ModernTerminologyValidator()

        assert validator.glossary == {}
        assert validator._term_index == {}

    def test_should_build_term_index_correctly(self, analyzer, sample_glossary):
        validator = analyzer

        assert "api" in validator._term_index
        assert "authentication" in validator._term_index
        assert "database" in validator._term_index

    def test_should_classify_common_english_word_rule_based(self, analyzer):
        validator = analyzer
        result = validator._classify_term_rule_based("the", None)

        assert result.term == "the"
        assert result.is_common_english is True
        assert result.is_domain_term is False

    def test_should_classify_acronym_as_domain_term(self, analyzer):
        validator = analyzer
        result = validator._classify_term_rule_based("API", None)

        assert result.term == "API"
        assert result.is_domain_term is True

    def test_should_classify_camelcase_as_domain_term(self, analyzer):
        validator = analyzer
        result = validator._classify_term_rule_based("AuthenticationService", None)

        assert result.term == "AuthenticationService"
        assert result.is_domain_term is True

    def test_should_classify_multi_word_term_as_domain_term(self, analyzer):
        validator = analyzer
        result = validator._classify_term_rule_based("user authentication", None)

        assert result.term == "user authentication"
        assert result.is_domain_term is True

    def test_should_detect_definition_with_is_pattern(self, analyzer):
        validator = analyzer
        context = "Authentication is the process of verifying user identity"
        result = validator._detect_definition_rule_based("Authentication", context)

        assert result.term == "Authentication"
        assert result.has_definition is True

    def test_should_detect_definition_with_means_pattern(self, analyzer):
        validator = analyzer
        context = "API means Application Programming Interface"
        result = validator._detect_definition_rule_based("API", context)

        assert result.term == "API"
        assert result.has_definition is True

    def test_should_not_detect_definition_when_not_present(self, analyzer):
        validator = analyzer
        context = "The API handles requests from clients"
        result = validator._detect_definition_rule_based("API", context)

        assert result.term == "API"
        assert result.has_definition is False
        assert result.confidence == 0.3

    def test_should_validate_defined_term(self, analyzer):
        validator = analyzer
        result = validator.validate_term_modern("API")

        assert result.term == "API"
        assert result.is_defined is True
        assert result.suggested_definition == "Application Programming Interface"

    def test_should_validate_undefined_domain_term(self, analyzer):
        validator = analyzer
        result = validator.validate_term_modern("AuthenticationService")

        assert result.term == "AuthenticationService"
        assert result.is_defined is False
        assert result.classification.is_domain_term is True
        assert result.classification.is_common_english is False

    def test_should_validate_common_english_term(self, analyzer):
        validator = analyzer
        result = validator.validate_term_modern("system")

        assert result.term == "system"
        assert result.is_defined is False
        assert result.classification.is_common_english is True
        assert result.classification.is_domain_term is False

    def test_should_validate_multiple_terms(self, analyzer):
        validator = analyzer
        terms = ["API", "system", "Blockchain", "the"]
        report = validator.validate_terms_modern(terms)

        assert isinstance(report, ModernValidationReport)
        assert report.statistics["total_terms"] == 4
        assert len(report.defined_terms) == 1
        assert len(report.undefined_terms) == 2
        assert len(report.common_english_terms) == 1

    def test_should_calculate_semantic_similarity_zero_when_no_model(self, analyzer):
        validator = analyzer
        similarity = validator._calculate_semantic_similarity("API", "interface")

        assert similarity == 0.0

    def test_should_extract_definition_from_context(self, analyzer):
        validator = analyzer
        context = "Authentication is the process of verifying identity"
        definition = validator._extract_definition_from_context(
            "Authentication", context
        )

        assert definition == "the process of verifying identity"

    def test_should_return_none_when_no_definition_extracted(self, analyzer):
        validator = analyzer
        context = "The system processes authentication requests"
        definition = validator._extract_definition_from_context(
            "Authentication", context
        )

        assert definition is None

    def test_should_create_empty_semantic_clusters_when_no_model(self, analyzer):
        validator = analyzer
        undefined_terms = [
            validator.validate_term_modern("term1"),
            validator.validate_term_modern("term2"),
        ]

        clusters = validator._create_semantic_clusters(undefined_terms)

        assert clusters == {}

    def test_should_get_priority_undefined_terms_empty_list(self, analyzer):
        validator = analyzer
        report = validator.validate_terms_modern([])

        assert isinstance(report, ModernValidationReport)
        assert report.statistics["total_terms"] == 0

    def test_should_get_priority_undefined_terms_sorted_by_score(self, analyzer):
        validator = analyzer

        result1 = validator.validate_term_modern("term1")
        result2 = validator.validate_term_modern("term2")

        priority_terms = validator.get_priority_undefined_terms(
            validator.validate_terms_modern([result1, result2]), top_k=2
        )

        assert len(priority_terms) == 2
        assert priority_terms[0]["term"] == "term1"
