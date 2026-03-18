"""
Concrete implementation of the ISemanticExtractor interface.

This module provides an implementation of the semantic extraction interface
using the existing SemanticExtractor class, ensuring compatibility with
the pipeline architecture while maintaining implementation independence.
"""

import time
from typing import List, Dict, Optional, Any

from .interfaces import (
    ISemanticExtractor, 
    AnalysisInput, 
    SemanticExtractionOutput,
    validate_interface_implementation
)
from .semantic_extractor import SemanticExtractor, SemanticProfile
from .document_processor import DocumentSegment


class SemanticExtractorImpl(ISemanticExtractor):
    """
    Concrete implementation of ISemanticExtractor using the existing SemanticExtractor.
    
    This class acts as an adapter between the interface contract and the existing
    SemanticExtractor implementation, ensuring independence while providing
    standardized input/output formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.extractor = SemanticExtractor()
        self.step_name = "SemanticExtractor"
        self.step_version = "1.0.0"
    
    def extract_from_text(self, text: str, **kwargs) -> SemanticExtractionOutput:
        """
        Extract semantic profiles from raw text.
        
        Args:
            text: The input text to analyze
            **kwargs: Additional parameters for extraction
            
        Returns:
            SemanticExtractionOutput containing extracted profiles and metadata
        """
        start_time = time.time()
        errors = []
        
        try:
            # Create a mock document segment for processing
            segment = DocumentSegment(
                element_type="paragraph",
                text=text,
                xpath="mock_xpath",
                page_number=1
            )
            
            # Extract profiles using the existing implementation
            profiles = self.extractor.extract_profiles_from_segment(segment)
            
            # Calculate extraction statistics
            stats = self._calculate_extraction_statistics(profiles)
            
            processing_time = time.time() - start_time
            
            return SemanticExtractionOutput(
                results=profiles,
                total_sentences=len(profiles),
                extraction_statistics=stats,
                metadata={
                    "input_length": len(text),
                    "config": self.config,
                    "extraction_method": "text_input"
                },
                confidence=self._calculate_overall_confidence(profiles),
                processing_time=processing_time,
                errors=errors if errors else None
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
                errors=errors
            )
    
    def extract_from_input(self, input_data: AnalysisInput) -> SemanticExtractionOutput:
        """
        Extract semantic profiles from standardized input.
        
        Args:
            input_data: Standardized analysis input
            
        Returns:
            SemanticExtractionOutput containing extracted profiles and metadata
        """
        start_time = time.time()
        
        # Extract from the content
        result = self.extract_from_text(
            input_data.content,
            source_id=input_data.source_id,
            content_type=input_data.content_type,
            **(input_data.metadata or {})
        )
        
        # Update metadata with input information
        if result.metadata:
            result.metadata.update({
                "source_id": input_data.source_id,
                "content_type": input_data.content_type,
                "original_metadata": input_data.metadata
            })
        
        return result
    
    def extract_single_profile(self, sentence: str) -> SemanticProfile:
        """
        Extract semantic profile from a single sentence.
        
        Args:
            sentence: Single sentence to analyze
            
        Returns:
            SemanticProfile for the sentence
        """
        return self.extractor.extract_semantic_profile(sentence)
    
    def get_supported_modalities(self) -> List[str]:
        """
        Get list of supported modality types.
        
        Returns:
            List of supported modality type names
        """
        from .semantic_extractor import ModalityType
        return [modality.value for modality in ModalityType]
    
    def validate_extraction_quality(self, profiles: List[SemanticProfile]) -> Dict[str, float]:
        """
        Validate the quality of extracted semantic profiles.
        
        Args:
            profiles: List of semantic profiles to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if not profiles:
            return {
                "avg_confidence": 0.0,
                "completeness_score": 0.0,
                "modality_coverage": 0.0,
                "overall_quality": 0.0
            }
        
        # Calculate average confidence
        avg_confidence = sum(p.confidence for p in profiles) / len(profiles)
        
        # Calculate completeness (percentage of profiles with key components)
        complete_profiles = sum(
            1 for p in profiles 
            if p.subject is not None and p.modality is not None
        )
        completeness_score = complete_profiles / len(profiles)
        
        # Calculate modality coverage
        profiles_with_modality = sum(1 for p in profiles if p.modality is not None)
        modality_coverage = profiles_with_modality / len(profiles)
        
        # Calculate overall quality score
        overall_quality = (avg_confidence + completeness_score + modality_coverage) / 3
        
        return {
            "avg_confidence": avg_confidence,
            "completeness_score": completeness_score,
            "modality_coverage": modality_coverage,
            "overall_quality": overall_quality
        }
    
    def get_step_name(self) -> str:
        """Get the name of this analysis step."""
        return self.step_name
    
    def get_step_version(self) -> str:
        """Get the version of this analysis step."""
        return self.step_version
    
    def is_healthy(self) -> bool:
        """Check if the step is healthy and ready to process data."""
        try:
            # Basic health check - try to extract from a simple sentence
            test_profile = self.extract_single_profile("The system shall process data.")
            return test_profile is not None
        except Exception:
            return False
    
    def _calculate_extraction_statistics(self, profiles: List[SemanticProfile]) -> Dict[str, int]:
        """Calculate statistics about the extraction results."""
        stats = {
            "total_profiles": len(profiles),
            "profiles_with_conditions": sum(1 for p in profiles if p.condition is not None),
            "profiles_with_subjects": sum(1 for p in profiles if p.subject is not None),
            "profiles_with_modalities": sum(1 for p in profiles if p.modality is not None),
            "profiles_with_objects": sum(1 for p in profiles if p.object is not None),
            "profiles_with_temporal": sum(1 for p in profiles if p.temporal is not None),
            "profiles_with_negation": sum(1 for p in profiles if p.negation is not None),
        }
        
        # Add modality-specific statistics
        from .semantic_extractor import ModalityType
        for modality_type in ModalityType:
            modality_profiles = self.extractor.get_profiles_by_modality(profiles, modality_type)
            stats[f"modality_{modality_type.value}"] = len(modality_profiles)
        
        return stats
    
    def _calculate_overall_confidence(self, profiles: List[SemanticProfile]) -> float:
        """Calculate overall confidence for the extraction."""
        if not profiles:
            return 0.0
        return sum(p.confidence for p in profiles) / len(profiles)


# Factory implementation
class SemanticExtractorFactory:
    """Factory for creating semantic extractor instances."""
    
    @staticmethod
    def create(config: Optional[Dict[str, Any]] = None) -> ISemanticExtractor:
        """
        Create a semantic extractor instance.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            ISemanticExtractor implementation
        """
        extractor = SemanticExtractorImpl(config)
        
        # Validate that the implementation is correct
        if not validate_interface_implementation(extractor, ISemanticExtractor):
            raise RuntimeError("SemanticExtractorImpl does not properly implement ISemanticExtractor")
        
        return extractor


# Register the implementation
def register_semantic_extractor():
    """Register the semantic extractor implementation."""
    # This function can be used to register the implementation with a dependency injection system
    return SemanticExtractorFactory
