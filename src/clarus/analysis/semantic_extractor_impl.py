import time
from typing import List, Dict, Optional, Any

from .interfaces import (
    ISemanticExtractor,
    AnalysisInput,
    SemanticExtractionOutput,
    validate_interface_implementation,
)
from .semantic_extractor import SemanticExtractor, SemanticProfile
from .document_processor import DocumentSegment


class SemanticExtractorImpl(ISemanticExtractor):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.extractor = SemanticExtractor()
        self.step_name = "SemanticExtractor"
        self.step_version = "1.0.0"

    def extract_from_text(self, text: str, **kwargs) -> SemanticExtractionOutput:
        start_time = time.time()
        errors = []

        try:
            segment = DocumentSegment(
                element_type="paragraph", text=text, xpath="mock_xpath", page_number=1
            )

            profiles = self.extractor.extract_profiles_from_segment(segment)

            stats = self._calculate_extraction_statistics(profiles)

            processing_time = time.time() - start_time

            return SemanticExtractionOutput(
                results=profiles,
                total_sentences=len(profiles),
                extraction_statistics=stats,
                metadata={
                    "input_length": len(text),
                    "config": self.config,
                    "extraction_method": "text_input",
                },
                confidence=self._calculate_overall_confidence(profiles),
                processing_time=processing_time,
                errors=errors if errors else None,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            errors.append(f"Extraction failed: {str(e)}")

            return SemanticExtractionOutput(
                results=[],
                total_sentences=0,
                extraction_statistics={},
                metadata={"error": True},
                confidence=0.0,
                processing_time=processing_time,
                errors=errors,
            )

    def extract_from_input(self, input_data: AnalysisInput) -> SemanticExtractionOutput:
        start_time = time.time()

        result = self.extract_from_text(
            input_data.content,
            source_id=input_data.source_id,
            content_type=input_data.content_type,
            **(input_data.metadata or {}),
        )

        if result.metadata:
            result.metadata.update(
                {
                    "source_id": input_data.source_id,
                    "content_type": input_data.content_type,
                    "original_metadata": input_data.metadata,
                }
            )

        return result

    def extract_single_profile(self, sentence: str) -> SemanticProfile:
        return self.extractor.extract_semantic_profile(sentence)

    def get_supported_modalities(self) -> List[str]:
        from .semantic_extractor import ModalityType

        return [modality.value for modality in ModalityType]

    def validate_extraction_quality(
        self, profiles: List[SemanticProfile]
    ) -> Dict[str, float]:
        if not profiles:
            return {
                "avg_confidence": 0.0,
                "completeness_score": 0.0,
                "modality_coverage": 0.0,
                "overall_quality": 0.0,
            }

        avg_confidence = sum(p.confidence for p in profiles) / len(profiles)

        complete_profiles = sum(
            1 for p in profiles if p.subject is not None and p.modality is not None
        )
        completeness_score = complete_profiles / len(profiles)

        profiles_with_modality = sum(1 for p in profiles if p.modality is not None)
        modality_coverage = profiles_with_modality / len(profiles)

        overall_quality = (avg_confidence + completeness_score + modality_coverage) / 3

        return {
            "avg_confidence": avg_confidence,
            "completeness_score": completeness_score,
            "modality_coverage": modality_coverage,
            "overall_quality": overall_quality,
        }

    def get_step_name(self) -> str:
        return self.step_name

    def get_step_version(self) -> str:
        return self.step_version

    def is_healthy(self) -> bool:
        try:
            test_profile = self.extract_single_profile("The system shall process data.")
            return test_profile is not None
        except Exception:
            return False

    def _calculate_extraction_statistics(
        self, profiles: List[SemanticProfile]
    ) -> Dict[str, int]:
        stats = {
            "total_profiles": len(profiles),
            "profiles_with_conditions": sum(
                1 for p in profiles if p.condition is not None
            ),
            "profiles_with_subjects": sum(1 for p in profiles if p.subject is not None),
            "profiles_with_modalities": sum(
                1 for p in profiles if p.modality is not None
            ),
            "profiles_with_objects": sum(1 for p in profiles if p.object is not None),
            "profiles_with_temporal": sum(
                1 for p in profiles if p.temporal is not None
            ),
            "profiles_with_negation": sum(
                1 for p in profiles if p.negation is not None
            ),
        }

        from .semantic_extractor import ModalityType

        for modality_type in ModalityType:
            modality_profiles = self.extractor.get_profiles_by_modality(
                profiles, modality_type
            )
            stats[f"modality_{modality_type.value}"] = len(modality_profiles)

        return stats

    def _calculate_overall_confidence(self, profiles: List[SemanticProfile]) -> float:
        if not profiles:
            return 0.0
        return sum(p.confidence for p in profiles) / len(profiles)


class SemanticExtractorFactory:

    @staticmethod
    def create(config: Optional[Dict[str, Any]] = None) -> ISemanticExtractor:
        extractor = SemanticExtractorImpl(config)

        if not validate_interface_implementation(extractor, ISemanticExtractor):
            raise RuntimeError(
                "SemanticExtractorImpl does not properly implement ISemanticExtractor"
            )

        return extractor


def register_semantic_extractor():
    return SemanticExtractorFactory
