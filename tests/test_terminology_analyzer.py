import pytest
from clarus.analysis.terminology_analyzer import (
    TerminologyAnalyzer,
    TermDefinition,
    TermVariant,
)


class TestTerminologyAnalyzer:

    @pytest.fixture
    def sample_glossary(self):
        return {
            "authentication": TermDefinition(
                preferred_form="authentication",
                definition="The process of verifying the identity of a user or system.",
                variants=[
                    TermVariant(text="authenticate", is_plural=False),
                    TermVariant(text="authenticated", is_plural=False),
                    TermVariant(text="auth", is_abbreviation=True),
                ],
            ),
            "API": TermDefinition(
                preferred_form="API",
                definition="Application Programming Interface",
                variants=[
                    TermVariant(
                        text="Application Programming Interface", is_acronym=True
                    )
                ],
            ),
        }

    @pytest.fixture
    def analyzer(self, sample_glossary):
        return TerminologyAnalyzer(glossary=sample_glossary)

    def test_should_initialize_analyzer_with_glossary(self, analyzer, sample_glossary):
        assert len(analyzer.glossary) == len(sample_glossary)
        assert "authentication" in analyzer.glossary
        assert "API" in analyzer.glossary

    def test_should_add_glossary_entries(self, analyzer):
        new_entry = TermDefinition(
            preferred_form="encryption",
            definition="The process of encoding information.",
            variants=[TermVariant(text="encrypt")],
        )

        analyzer.add_glossary_entries([new_entry])

        assert "encryption" in analyzer.glossary
        assert (
            analyzer.glossary["encryption"].definition
            == "The process of encoding information."
        )

    def test_should_extract_candidate_terms(self, analyzer):
        text = "The API requires secure authentication using OAuth 2.0."
        candidates = analyzer._extract_candidate_terms(text)

        expected_terms = ["API", "secure authentication", "OAuth 2.0", "OAuth"]
        for term in expected_terms:
            assert term in candidates

    def test_should_analyze_document_defined_terms(self, analyzer):
        text = """
        The system uses API for authentication. Users must authenticate
        using their credentials. The API supports OAuth 2.0.
        """

        analysis = analyzer.analyze_document(text)

        assert analysis.statistics["defined_terms"] >= 2  # API and authentication
        assert any(occ.term.lower() == "api" for occ in analysis.term_occurrences)
        assert any(
            "authentication" in occ.term.lower() for occ in analysis.term_occurrences
        )

    def test_should_analyze_document_undefined_terms(self, analyzer):
        """Test identification of undefined terms"""
        text = "The system uses JWT tokens for session management."

        analysis = analyzer.analyze_document(text)

        assert analysis.statistics["undefined_terms"] > 0
        assert any("JWT" in occ.term for occ in analysis.undefined_terms)
        assert any(
            "session management" in occ.term.lower() for occ in analysis.undefined_terms
        )

    def test_should_extract_definitions(self, analyzer):
        text = """
        OAuth 2.0 is an authorization framework.
        JWT (JSON Web Token) is a compact token format.
        """

        definitions = analyzer._extract_definitions(text)

        assert len(definitions) >= 2
        assert "oauth 2.0" in [k.lower() for k in definitions.keys()]
        assert "jwt" in [k.lower() for k in definitions.keys()]

        jwt_def = next(v for k, v in definitions.items() if k.lower() == "jwt")
        assert "JSON Web Token" in jwt_def.definition

    def test_should_suggest_term_variants(self, analyzer):
        term = "access token"
        variants = analyzer.suggest_term_variants(term)

        variant_texts = [v.text.lower() for v in variants]
        assert len(variants) >= 3  # Original, plural, and at least one other variant
        assert "access token" in variant_texts
        assert "access tokens" in variant_texts
        assert "access token's" in variant_texts

        assert any(v.is_acronym for v in variants)
        assert any(v.text == "AT" for v in variants)

    def test_should_find_similar_terms(self, analyzer):
        similar_terms = [
            TermDefinition("authentication", "Verification of identity", []),
            TermDefinition("authorization", "Permission granting", []),
            TermDefinition("authenticate", "To verify identity", []),
        ]
        analyzer.add_glossary_entries(similar_terms)

        similar = analyzer.find_similar_terms("auth")

        assert len(similar) >= 2
        similar_terms = [term.lower() for term, _ in similar]
        assert "authentication" in similar_terms
        assert "authenticate" in similar_terms

    def test_should_calculate_term_frequency(self, analyzer):
        text = """
        The API requires authentication. The authentication process
        uses tokens. The API is stateless.
        """

        analysis = analyzer.analyze_document(text)
        freq = analysis.get_term_frequency()

        assert freq.get("api", 0) == 2
        assert freq.get("authentication", 0) == 2
        assert freq.get("process", 0) == 1

    def test_should_rank_undefined_terms(self, analyzer):
        text = """
        The system uses JWT tokens. JWT tokens are secure.
        The session timeout is 30 minutes. Session management is important.
        """

        analysis = analyzer.analyze_document(text)
        undefined = analysis.get_undefined_terms()

        assert len(undefined) >= 2
        assert (
            undefined[0][1] >= undefined[-1][1]
        )  # First has higher or equal frequency

        undefined_terms = [term.lower() for term, _ in undefined]
        assert "jwt tokens" in undefined_terms or "jwt" in undefined_terms
        assert "session timeout" in [t.lower() for t, _ in undefined] or "timeout" in [
            t.lower() for t, _ in undefined
        ]

    def test_should_identify_potential_terms(self, analyzer):
        assert analyzer._is_potential_term("API") is True
        assert analyzer._is_potential_term("authentication") is True
        assert analyzer._is_potential_term("OAuth 2.0") is True

        assert analyzer._is_potential_term("the") is False  # Stopword
        assert analyzer._is_potential_term("123") is False  # Just numbers
        assert analyzer._is_potential_term("a") is False  # Too short

    def test_should_split_into_sentences(self, analyzer):
        text = "This is a test. This is another test! And another one?"
        sentences = analyzer._split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0].endswith("test.")
        assert sentences[1].endswith("test!")
        assert sentences[2].endswith("one?")

    def test_should_analyze_document_with_definitions(self, analyzer):
        text = """
        OAuth 2.0 is an authorization framework. JWT (JSON Web Token)
        is a compact token format. The system uses both.
        """

        analysis = analyzer.analyze_document(text)

        assert analysis.statistics["new_definitions"] >= 2
        assert any("OAuth 2.0" in d for d in analysis.defined_terms)
        assert any("JWT" in d for d in analysis.defined_terms)

        assert any(occ.term == "OAuth 2.0" for occ in analysis.term_occurrences)
        assert any(occ.term == "JWT" for occ in analysis.term_occurrences)
