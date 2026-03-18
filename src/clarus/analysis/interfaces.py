"""
Interfaces for the CLARUS analysis pipeline.

This module defines abstract base classes that ensure independent implementations
of each analysis step while maintaining clear contracts between components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .semantic_extractor import SemanticProfile, SemanticAnchor
from .s2_taxonomy import ErrorType, ErrorDefinition


@dataclass
class AnalysisInput:
    """Standardized input format for analysis components."""
    
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source_id: Optional[str] = None
    content_type: Optional[str] = None  # e.g., "requirement", "specification", "procedure"


@dataclass
class AnalysisOutput:
    """Standardized output format for analysis components."""
    
    results: Any
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: Optional[float] = None
    errors: Optional[List[str]] = None


@dataclass
class SemanticExtractionOutput(AnalysisOutput):
    """Output format for semantic extraction."""
    
    results: List[SemanticProfile]
    total_sentences: int
    extraction_statistics: Optional[Dict[str, int]] = None


@dataclass
class S2AnalysisOutput(AnalysisOutput):
    """Output format for S2 Taxonomy analysis."""
    
    results: List[Tuple[ErrorType, List[str]]]
    error_definitions: Dict[ErrorType, ErrorDefinition]
    severity_summary: Dict[str, int]
    category_summary: Dict[str, int]


class ISemanticExtractor(ABC):
    """
    Interface for semantic extraction components.
    
    This interface defines the contract for extracting semantic information
    from text, ensuring that different implementations can be swapped
    without affecting downstream components.
    """
    
    @abstractmethod
    def extract_from_text(self, text: str, **kwargs) -> SemanticExtractionOutput:
        """
        Extract semantic profiles from raw text.
        
        Args:
            text: The input text to analyze
            **kwargs: Additional parameters for extraction
            
        Returns:
            SemanticExtractionOutput containing extracted profiles and metadata
        """
        pass
    
    @abstractmethod
    def extract_from_input(self, input_data: AnalysisInput) -> SemanticExtractionOutput:
        """
        Extract semantic profiles from standardized input.
        
        Args:
            input_data: Standardized analysis input
            
        Returns:
            SemanticExtractionOutput containing extracted profiles and metadata
        """
        pass
    
    @abstractmethod
    def extract_single_profile(self, sentence: str) -> SemanticProfile:
        """
        Extract semantic profile from a single sentence.
        
        Args:
            sentence: Single sentence to analyze
            
        Returns:
            SemanticProfile for the sentence
        """
        pass
    
    @abstractmethod
    def get_supported_modalities(self) -> List[str]:
        """
        Get list of supported modality types.
        
        Returns:
            List of supported modality type names
        """
        pass
    
    @abstractmethod
    def validate_extraction_quality(self, profiles: List[SemanticProfile]) -> Dict[str, float]:
        """
        Validate the quality of extracted semantic profiles.
        
        Args:
            profiles: List of semantic profiles to validate
            
        Returns:
            Dictionary with quality metrics
        """
        pass


class IS2TaxonomyAnalyzer(ABC):
    """
    Interface for S2 Taxonomy analysis components.
    
    This interface defines the contract for analyzing text according to
    the S2 Taxonomy framework, enabling different implementations while
    maintaining consistent output formats.
    """
    
    @abstractmethod
    def analyze_text(self, text: str, **kwargs) -> S2AnalysisOutput:
        """
        Analyze text for S2 Taxonomy errors.
        
        Args:
            text: The input text to analyze
            **kwargs: Additional parameters for analysis
            
        Returns:
            S2AnalysisOutput containing detected errors and metadata
        """
        pass
    
    @abstractmethod
    def analyze_semantic_profiles(self, profiles: List[SemanticProfile]) -> S2AnalysisOutput:
        """
        Analyze semantic profiles for S2 Taxonomy errors.
        
        Args:
            profiles: List of semantic profiles to analyze
            
        Returns:
            S2AnalysisOutput containing detected errors and metadata
        """
        pass
    
    @abstractmethod
    def analyze_from_input(self, input_data: AnalysisInput) -> S2AnalysisOutput:
        """
        Analyze standardized input for S2 Taxonomy errors.
        
        Args:
            input_data: Standardized analysis input
            
        Returns:
            S2AnalysisOutput containing detected errors and metadata
        """
        pass
    
    @abstractmethod
    def get_error_definition(self, error_type: ErrorType) -> ErrorDefinition:
        """
        Get detailed information about a specific error type.
        
        Args:
            error_type: The error type to get details for
            
        Returns:
            ErrorDefinition with full details
        """
        pass
    
    @abstractmethod
    def get_supported_error_types(self) -> List[ErrorType]:
        """
        Get list of supported error types.
        
        Returns:
            List of supported ErrorType enum values
        """
        pass
    
    @abstractmethod
    def calculate_reliability_metrics(self, 
                                     human_annotations: List[Tuple[str, ErrorType]],
                                     system_predictions: List[Tuple[str, ErrorType]]) -> Dict[str, float]:
        """
        Calculate reliability metrics for the taxonomy analysis.
        
        Args:
            human_annotations: List of (text, error_type) tuples from human annotators
            system_predictions: List of (text, error_type) tuples from system
            
        Returns:
            Dictionary containing reliability metrics including Cohen's Kappa
        """
        pass


class IAnalysisPipeline(ABC):
    """
    Interface for the complete analysis pipeline.
    
    This interface defines the contract for orchestrating multiple
    analysis steps while maintaining independence between components.
    """
    
    @abstractmethod
    def process_document(self, document: AnalysisInput) -> Dict[str, AnalysisOutput]:
        """
        Process a complete document through all analysis steps.
        
        Args:
            document: The document to process
            
        Returns:
            Dictionary mapping step names to their respective outputs
        """
        pass
    
    @abstractmethod
    def process_semantic_profiles(self, profiles: List[SemanticProfile]) -> S2AnalysisOutput:
        """
        Process pre-extracted semantic profiles through S2 analysis.
        
        Args:
            profiles: List of semantic profiles to analyze
            
        Returns:
            S2AnalysisOutput with analysis results
        """
        pass
    
    @abstractmethod
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline performance and configuration.
        
        Returns:
            Dictionary with pipeline statistics
        """
        pass
    
    @abstractmethod
    def validate_pipeline_health(self) -> Dict[str, bool]:
        """
        Validate the health and configuration of all pipeline components.
        
        Returns:
            Dictionary with health check results for each component
        """
        pass


class IAnalysisStep(ABC):
    """
    Base interface for individual analysis steps.
    
    This provides a common contract that all analysis steps must follow,
    ensuring they can be orchestrated by the pipeline.
    """
    
    @abstractmethod
    def process(self, input_data: AnalysisInput) -> AnalysisOutput:
        """
        Process input data according to the step's specific logic.
        
        Args:
            input_data: Standardized analysis input
            
        Returns:
            AnalysisOutput with processing results
        """
        pass
    
    @abstractmethod
    def get_step_name(self) -> str:
        """
        Get the name of this analysis step.
        
        Returns:
            String name of the step
        """
        pass
    
    @abstractmethod
    def get_step_version(self) -> str:
        """
        Get the version of this analysis step.
        
        Returns:
            String version identifier
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the step is healthy and ready to process data.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


# Factory interfaces for creating analysis components
class ISemanticExtractorFactory(ABC):
    """Factory interface for creating semantic extractors."""
    
    @abstractmethod
    def create_extractor(self, config: Optional[Dict[str, Any]] = None) -> ISemanticExtractor:
        """Create a semantic extractor instance."""
        pass


class IS2TaxonomyAnalyzerFactory(ABC):
    """Factory interface for creating S2 taxonomy analyzers."""
    
    @abstractmethod
    def create_analyzer(self, config: Optional[Dict[str, Any]] = None) -> IS2TaxonomyAnalyzer:
        """Create an S2 taxonomy analyzer instance."""
        pass


class IAnalysisPipelineFactory(ABC):
    """Factory interface for creating analysis pipelines."""
    
    @abstractmethod
    def create_pipeline(self, 
                       semantic_extractor: ISemanticExtractor,
                       s2_analyzer: IS2TaxonomyAnalyzer,
                       config: Optional[Dict[str, Any]] = None) -> IAnalysisPipeline:
        """Create an analysis pipeline with the specified components."""
        pass


# Utility functions for working with interfaces
def validate_interface_implementation(instance: Any, interface_class: type) -> bool:
    """
    Validate that an instance properly implements an interface.
    
    Args:
        instance: The instance to validate
        interface_class: The interface class to check against
        
    Returns:
        True if instance properly implements the interface
    """
    if not isinstance(instance, interface_class):
        return False
    
    # Check that all abstract methods are implemented
    for method_name in interface_class.__abstractmethods__:
        if not hasattr(instance, method_name):
            return False
        method = getattr(instance, method_name)
        if not callable(method):
            return False
    
    return True


def create_standard_input(content: str, 
                         source_id: Optional[str] = None,
                         content_type: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> AnalysisInput:
    """
    Create a standardized AnalysisInput object.
    
    Args:
        content: The text content to analyze
        source_id: Optional identifier for the source
        content_type: Optional type of content
        metadata: Optional additional metadata
        
    Returns:
        AnalysisInput object with the provided data
    """
    return AnalysisInput(
        content=content,
        metadata=metadata or {},
        source_id=source_id,
        content_type=content_type
    )
