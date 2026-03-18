"""
S2 Taxonomy for Technical Writing Error Classification

This module defines the S2 Taxonomy framework for identifying and categorizing
errors in technical documentation, particularly focusing on clarity, consistency,
and logical coherence issues.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re


class ErrorType(Enum):
    """Enumeration of all error types in the S2 Taxonomy."""
    
    UNDEFINED_TERM = "FAIL-001"
    SYNONYM_INCONSISTENCY = "FAIL-002"
    DIRECT_CONTRADICTION = "FAIL-003"
    MODAL_INCONSISTENCY = "FAIL-004"
    HIDDEN_NORMATIVE = "FAIL-005"
    VAGUE_QUALIFIERS = "FAIL-006"
    AMBIGUOUS_REFERENT = "FAIL-007"
    MISSING_TEMPORAL_ANCHOR = "FAIL-008"


class ErrorCategory(Enum):
    """Categories of errors in the S2 Taxonomy."""
    
    TERMINOLOGY = "Terminology"
    LOGICAL = "Logical"
    VAGUENESS = "Vagueness"
    CONTEXT = "Context"


class PerceivedSeverity(Enum):
    """Severity levels for errors."""
    
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class ErrorDefinition:
    """Definition of an error type in the S2 Taxonomy."""
    
    uid: str
    error_type: ErrorType
    category: ErrorCategory
    sample_text: str
    description: str
    severity: PerceivedSeverity


class S2Taxonomy:
    """
    Main class for the S2 Taxonomy framework.
    
    Provides classification and validation capabilities for technical writing errors.
    """
    
    def __init__(self):
        self.error_definitions = self._initialize_error_definitions()
        self.patterns = self._initialize_patterns()
    
    def _initialize_error_definitions(self) -> Dict[ErrorType, ErrorDefinition]:
        """Initialize all error definitions from the S2 Taxonomy."""
        return {
            ErrorType.UNDEFINED_TERM: ErrorDefinition(
                uid="FAIL-001",
                error_type=ErrorType.UNDEFINED_TERM,
                category=ErrorCategory.TERMINOLOGY,
                sample_text="The system shall process the data.",
                description="Without a clear definition, \"data\" and \"process\" are ambiguous.",
                severity=PerceivedSeverity.HIGH
            ),
            ErrorType.SYNONYM_INCONSISTENCY: ErrorDefinition(
                uid="FAIL-002",
                error_type=ErrorType.SYNONYM_INCONSISTENCY,
                category=ErrorCategory.TERMINOLOGY,
                sample_text="The Operator shall monitor the gauge. The Technician must log the results.",
                description="Creates \"Who\" ambiguity; are these intended to be different people, or is this inconsistent writing?",
                severity=PerceivedSeverity.MEDIUM
            ),
            ErrorType.DIRECT_CONTRADICTION: ErrorDefinition(
                uid="FAIL-003",
                error_type=ErrorType.DIRECT_CONTRADICTION,
                category=ErrorCategory.LOGICAL,
                sample_text="Staff must enter. Staff may not enter.",
                description="May create an unenforceable rule when scopes overlap.",
                severity=PerceivedSeverity.HIGH
            ),
            ErrorType.MODAL_INCONSISTENCY: ErrorDefinition(
                uid="FAIL-004",
                error_type=ErrorType.MODAL_INCONSISTENCY,
                category=ErrorCategory.LOGICAL,
                sample_text="Users must authenticate. Users may bypass authentication.",
                description="May create a loophole for complying with the rule.",
                severity=PerceivedSeverity.MEDIUM
            ),
            ErrorType.HIDDEN_NORMATIVE: ErrorDefinition(
                uid="FAIL-005",
                error_type=ErrorType.HIDDEN_NORMATIVE,
                category=ErrorCategory.CONTEXT,
                sample_text="For instance, one should always verify credentials.",
                description="If this is found in an \"Informative\" section, this can accidentally creates a rule in a section that may not be referenced as often.",
                severity=PerceivedSeverity.MEDIUM
            ),
            ErrorType.VAGUE_QUALIFIERS: ErrorDefinition(
                uid="FAIL-006",
                error_type=ErrorType.VAGUE_QUALIFIERS,
                category=ErrorCategory.VAGUENESS,
                sample_text="When the pressure is too high, the valve shall open.",
                description="\"Too high\" is non-verifiable without a specific value.",
                severity=PerceivedSeverity.LOW
            ),
            ErrorType.AMBIGUOUS_REFERENT: ErrorDefinition(
                uid="FAIL-007",
                error_type=ErrorType.AMBIGUOUS_REFERENT,
                category=ErrorCategory.VAGUENESS,
                sample_text="The system connects to the server. It must be secure.",
                description="\"It\" could refer to a system or a server.",
                severity=PerceivedSeverity.MEDIUM
            ),
            ErrorType.MISSING_TEMPORAL_ANCHOR: ErrorDefinition(
                uid="FAIL-008",
                error_type=ErrorType.MISSING_TEMPORAL_ANCHOR,
                category=ErrorCategory.LOGICAL,
                sample_text="The system shall reboot as soon as possible.",
                description="No measurable deadline for compliance.",
                severity=PerceivedSeverity.HIGH
            )
        }
    
    def _initialize_patterns(self) -> Dict[ErrorType, List[re.Pattern]]:
        """Initialize regex patterns for detecting errors."""
        return {
            ErrorType.UNDEFINED_TERM: [
                re.compile(r'\b(system|data|process|interface|function|module)\b', re.IGNORECASE)
            ],
            ErrorType.SYNONYM_INCONSISTENCY: [
                re.compile(r'\b(operator|technician|user|staff|personnel|admin)\b', re.IGNORECASE)
            ],
            ErrorType.DIRECT_CONTRADICTION: [
                re.compile(r'\b(must|shall|required|mandatory)\b.*\b(may not|shall not|must not|prohibited)\b', re.IGNORECASE),
                re.compile(r'\b(may|can|optional)\b.*\b(must|shall|required|mandatory)\b', re.IGNORECASE)
            ],
            ErrorType.MODAL_INCONSISTENCY: [
                re.compile(r'\b(must|shall|required)\b.*\b(may|can|optional)\b', re.IGNORECASE)
            ],
            ErrorType.HIDDEN_NORMATIVE: [
                re.compile(r'\b(always|never|must|shall|required|should)\b.*\b(for example|for instance|e\.g\.|such as)\b', re.IGNORECASE)
            ],
            ErrorType.VAGUE_QUALIFIERS: [
                re.compile(r'\b(too|very|quite|rather|somewhat|approximately|about|generally|typically)\s+\w+\b', re.IGNORECASE)
            ],
            ErrorType.AMBIGUOUS_REFERENT: [
                re.compile(r'\b(it|this|that|they|them)\s+(must|shall|should|will|can|may)\b', re.IGNORECASE)
            ],
            ErrorType.MISSING_TEMPORAL_ANCHOR: [
                re.compile(r'\b(as soon as possible|promptly|quickly|soon|eventually|in due course)\b', re.IGNORECASE)
            ]
        }
    
    def get_error_definition(self, error_type: ErrorType) -> ErrorDefinition:
        """Get the definition for a specific error type."""
        return self.error_definitions[error_type]
    
    def get_all_error_definitions(self) -> List[ErrorDefinition]:
        """Get all error definitions."""
        return list(self.error_definitions.values())
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorDefinition]:
        """Get all errors in a specific category."""
        return [defn for defn in self.error_definitions.values() if defn.category == category]
    
    def get_errors_by_severity(self, severity: PerceivedSeverity) -> List[ErrorDefinition]:
        """Get all errors with a specific severity level."""
        return [defn for defn in self.error_definitions.values() if defn.severity == severity]
    
    def detect_errors(self, text: str) -> List[Tuple[ErrorType, List[str]]]:
        """
        Detect potential errors in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of tuples containing (error_type, matching_phrases)
        """
        detected_errors = []
        
        for error_type, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                found_matches = pattern.findall(text)
                matches.extend(found_matches)
            
            if matches:
                detected_errors.append((error_type, list(set(matches))))  # Remove duplicates
        
        return detected_errors
    
    def calculate_cohens_kappa(self, observed_agreement: float, chance_agreement: float) -> float:
        """
        Calculate Cohen's Kappa for inter-rater reliability.
        
        Args:
            observed_agreement: Proportion of actual agreement (Pₒ)
            chance_agreement: Proportion of chance agreement (Pₑ)
            
        Returns:
            Cohen's Kappa value
        """
        if chance_agreement == 1.0:
            return 0.0  # Avoid division by zero
        
        kappa = (observed_agreement - chance_agreement) / (1 - chance_agreement)
        return kappa
    
    def validate_taxonomy_completeness(self) -> Dict[str, bool]:
        """
        Validate that the taxonomy is complete and properly structured.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check all error types have definitions
        validation_results['all_types_defined'] = all(
            error_type in self.error_definitions for error_type in ErrorType
        )
        
        # Check all categories have at least one error
        categories_used = set(defn.category for defn in self.error_definitions.values())
        validation_results['all_categories_used'] = all(
            category in categories_used for category in ErrorCategory
        )
        
        # Check all severities are used
        severities_used = set(defn.severity for defn in self.error_definitions.values())
        validation_results['all_severities_used'] = all(
            severity in severities_used for severity in PerceivedSeverity
        )
        
        # Check all error types have detection patterns
        validation_results['all_types_have_patterns'] = all(
            error_type in self.patterns for error_type in ErrorType
        )
        
        return validation_results
    
    def get_taxonomy_summary(self) -> Dict[str, int]:
        """
        Get a summary of the taxonomy structure.
        
        Returns:
            Dictionary with counts by category and severity
        """
        summary = {
            'total_errors': len(self.error_definitions),
            'by_category': {},
            'by_severity': {}
        }
        
        for defn in self.error_definitions.values():
            # Count by category
            category_name = defn.category.value
            summary['by_category'][category_name] = summary['by_category'].get(category_name, 0) + 1
            
            # Count by severity
            severity_name = defn.severity.value
            summary['by_severity'][severity_name] = summary['by_severity'].get(severity_name, 0) + 1
        
        return summary


# Convenience functions for common operations
def get_s2_taxonomy() -> S2Taxonomy:
    """Get an instance of the S2 Taxonomy."""
    return S2Taxonomy()


def analyze_text_for_errors(text: str) -> List[Tuple[ErrorType, List[str]]]:
    """
    Analyze text for S2 Taxonomy errors.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of detected errors with matching phrases
    """
    taxonomy = get_s2_taxonomy()
    return taxonomy.detect_errors(text)


def get_error_details(error_type: ErrorType) -> ErrorDefinition:
    """
    Get detailed information about a specific error type.
    
    Args:
        error_type: The error type to get details for
        
    Returns:
        ErrorDefinition with full details
    """
    taxonomy = get_s2_taxonomy()
    return taxonomy.get_error_definition(error_type)
