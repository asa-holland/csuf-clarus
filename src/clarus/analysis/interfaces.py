from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .semantic_extractor import SemanticProfile, SemanticAnchor
from .s2_taxonomy import ErrorType, ErrorDefinition


@dataclass
class AnalysisInput:
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source_id: Optional[str] = None
    content_type: Optional[str] = None


@dataclass
class AnalysisOutput:
    results: Any
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: Optional[float] = None
    errors: Optional[List[str]] = None


@dataclass
class SemanticExtractionOutput(AnalysisOutput):
    results: List[SemanticProfile]
    total_sentences: int
    extraction_statistics: Optional[Dict[str, int]] = None


@dataclass
class S2AnalysisOutput(AnalysisOutput):
    results: List[Tuple[ErrorType, List[str]]]
    error_definitions: Dict[ErrorType, ErrorDefinition]
    severity_summary: Dict[str, int]
    category_summary: Dict[str, int]


class ISemanticExtractor(ABC):
    @abstractmethod
    def extract_from_text(self, text: str, **kwargs) -> SemanticExtractionOutput:
        pass

    @abstractmethod
    def extract_from_input(self, input_data: AnalysisInput) -> SemanticExtractionOutput:
        pass

    @abstractmethod
    def extract_single_profile(self, sentence: str) -> SemanticProfile:
        pass

    @abstractmethod
    def get_supported_modalities(self) -> List[str]:
        pass

    @abstractmethod
    def validate_extraction_quality(
        self, profiles: List[SemanticProfile]
    ) -> Dict[str, float]:
        pass


class IS2TaxonomyAnalyzer(ABC):

    @abstractmethod
    def analyze_text(self, text: str, **kwargs) -> S2AnalysisOutput:
        pass

    @abstractmethod
    def analyze_semantic_profiles(
        self, profiles: List[SemanticProfile]
    ) -> S2AnalysisOutput:
        pass

    @abstractmethod
    def analyze_from_input(self, input_data: AnalysisInput) -> S2AnalysisOutput:
        pass

    @abstractmethod
    def get_error_definition(self, error_type: ErrorType) -> ErrorDefinition:
        pass

    @abstractmethod
    def get_supported_error_types(self) -> List[ErrorType]:
        pass

    @abstractmethod
    def calculate_reliability_metrics(
        self,
        human_annotations: List[Tuple[str, ErrorType]],
        system_predictions: List[Tuple[str, ErrorType]],
    ) -> Dict[str, float]:
        pass


class IAnalysisPipeline(ABC):

    @abstractmethod
    def process_document(self, document: AnalysisInput) -> Dict[str, AnalysisOutput]:
        pass

    @abstractmethod
    def process_semantic_profiles(
        self, profiles: List[SemanticProfile]
    ) -> S2AnalysisOutput:
        pass

    @abstractmethod
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate_pipeline_health(self) -> Dict[str, bool]:
        pass


class IAnalysisStep(ABC):

    @abstractmethod
    def process(self, input_data: AnalysisInput) -> AnalysisOutput:
        pass

    @abstractmethod
    def get_step_name(self) -> str:
        pass

    @abstractmethod
    def get_step_version(self) -> str:
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        pass


class ISemanticExtractorFactory(ABC):

    @abstractmethod
    def create_extractor(
        self, config: Optional[Dict[str, Any]] = None
    ) -> ISemanticExtractor:
        pass


class IS2TaxonomyAnalyzerFactory(ABC):

    @abstractmethod
    def create_analyzer(
        self, config: Optional[Dict[str, Any]] = None
    ) -> IS2TaxonomyAnalyzer:
        """Create an S2 taxonomy analyzer instance."""
        pass


class IAnalysisPipelineFactory(ABC):

    @abstractmethod
    def create_pipeline(
        self,
        semantic_extractor: ISemanticExtractor,
        s2_analyzer: IS2TaxonomyAnalyzer,
        config: Optional[Dict[str, Any]] = None,
    ) -> IAnalysisPipeline:
        """Create an analysis pipeline with the specified components."""
        pass


def validate_interface_implementation(instance: Any, interface_class: type) -> bool:
    if not isinstance(instance, interface_class):
        return False

    for method_name in interface_class.__abstractmethods__:
        if not hasattr(instance, method_name):
            return False
        method = getattr(instance, method_name)
        if not callable(method):
            return False

    return True


def create_standard_input(
    content: str,
    source_id: Optional[str] = None,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AnalysisInput:
    return AnalysisInput(
        content=content,
        metadata=metadata or {},
        source_id=source_id,
        content_type=content_type,
    )
