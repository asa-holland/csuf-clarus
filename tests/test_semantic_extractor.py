from clarus.analysis.semantic_extractor import (
    SemanticExtractor,
    SemanticProfile,
    ModalityType,
)
from clarus.analysis.document_processor import DocumentSegment, ElementType


class TestSemanticExtractor:

    def setup_method(self):
        self.extractor = SemanticExtractor()

    def test_extract_modality_obligation(self):
        text = "The system shall process the data."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.modality is not None
        assert "shall" in profile.modality.text.lower()
        assert profile.modality.anchor_type == "modality"
        assert profile.modality.confidence >= 0.8

    def test_extract_modality_prohibition(self):
        text = "Users must not share their credentials."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.modality is not None
        assert "must not" in profile.modality.text.lower()
        assert profile.modality.anchor_type == "modality"

    def test_extract_condition(self):
        text = "If the temperature exceeds 90°C, the system shall shut down."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.condition is not None
        assert "temperature exceeds 90°C" in profile.condition.text
        assert profile.condition.anchor_type == "condition"

    def test_extract_subject(self):
        text = "The operator must verify the readings."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.subject is not None
        assert "operator" in profile.subject.text.lower()
        assert profile.subject.anchor_type == "subject"

    def test_extract_object(self):
        text = "The system shall encrypt all user data."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.object is not None
        assert "user data" in profile.object.text.lower()
        assert profile.object.anchor_type == "object"

    def test_extract_temporal(self):
        text = "The system must respond within 2 seconds."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.temporal is not None
        assert "within 2 seconds" in profile.temporal.text.lower()
        assert profile.temporal.anchor_type == "temporal"

    def test_extract_negation(self):
        text = "The system shall not store passwords in plaintext."
        profile = self.extractor.extract_semantic_profile(text)

        assert profile.negation is not None
        assert profile.negation.text.lower() == "not"
        assert profile.negation.anchor_type == "negation"

    def test_extract_profiles_from_segment(self):
        segment = DocumentSegment(
            text="""1. Requirements
1.1 The system shall authenticate users.
1.2 Users must change passwords every 90 days.""",
            element_type=ElementType.NORMATIVE,
        )

        profiles = self.extractor.extract_profiles_from_segment(segment)

        assert len(profiles) >= 2  # At least one profile per sentence
        assert all(isinstance(p, SemanticProfile) for p in profiles)
        assert profiles[0].metadata["segment_type"] == "normative"

    def test_profile_confidence_calculation(self):
        text1 = "The system shall encrypt all data."
        profile1 = self.extractor.extract_semantic_profile(text1)

        text2 = "Process the data."
        profile2 = self.extractor.extract_semantic_profile(text2)

        assert profile1.confidence > profile2.confidence
        assert 0.0 <= profile1.confidence <= 1.0
        assert 0.0 <= profile2.confidence <= 1.0

    def test_get_profiles_by_modality(self):
        profiles = [
            self.extractor.extract_semantic_profile("The system shall start."),
            self.extractor.extract_semantic_profile("Users may log in."),
            self.extractor.extract_semantic_profile("The process must complete."),
        ]

        obligations = self.extractor.get_profiles_by_modality(
            profiles, ModalityType.OBLIGATION
        )

        assert len(obligations) >= 2  # "shall" and "must" are both obligations
        assert all(
            any(p in profile.modality.text.lower() for p in ["shall", "must"])
            for profile in obligations
        )

    def test_find_profiles_with_conditions(self):
        profiles = [
            self.extractor.extract_semantic_profile("If error occurs, log it."),
            self.extractor.extract_semantic_profile("The system shall start."),
            self.extractor.extract_semantic_profile("When ready, notify user."),
        ]

        conditional_profiles = self.extractor.find_profiles_with_conditions(profiles)

        assert len(conditional_profiles) == 2  # "If" and "When" conditions
        assert all(p.condition is not None for p in conditional_profiles)

    def test_empty_input(self):
        profile = self.extractor.extract_semantic_profile("")

        assert isinstance(profile, SemanticProfile)
        assert profile.original_sentence == ""
        assert profile.confidence == 0.0
        assert all(
            getattr(profile, attr) is None
            for attr in [
                "condition",
                "subject",
                "modality",
                "object",
                "temporal",
                "negation",
            ]
        )

    def test_complex_sentence_parsing(self):
        text = """If the temperature exceeds 90°C, the system shall immediately
        activate the cooling fans and notify the operator, but it must not
        shut down unless absolutely necessary."""

        profile = self.extractor.extract_semantic_profile(text)

        assert profile.condition is not None
        assert "temperature exceeds 90°C" in profile.condition.text

        assert profile.modality is not None
        assert "shall" in profile.modality.text.lower()

        assert profile.temporal is not None
        assert "immediately" in profile.temporal.text.lower()

        assert profile.negation is not None
        assert "not" in profile.negation.text.lower()

        assert 0.5 <= profile.confidence <= 1.0
