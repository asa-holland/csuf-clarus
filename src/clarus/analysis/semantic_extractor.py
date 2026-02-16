import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .document_processor import DocumentSegment


class ModalityType(Enum):

    OBLIGATION = "obligation"  # shall, must, is required to
    PROHIBITION = "prohibition"  # shall not, must not, is prohibited from
    PERMISSION = "permission"  # may, can, is permitted to
    RECOMMENDATION = "recommendation"  # should, ought to
    CAPABILITY = "capability"  # can, is able to
    NONE = "none"  # no clear modality


@dataclass
class SemanticAnchor:
    anchor_type: str
    text: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class SemanticProfile:
    original_sentence: str
    condition: Optional[SemanticAnchor] = None
    subject: Optional[SemanticAnchor] = None
    modality: Optional[SemanticAnchor] = None
    object: Optional[SemanticAnchor] = None
    temporal: Optional[SemanticAnchor] = None
    negation: Optional[SemanticAnchor] = None
    confidence: float = 0.0
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticExtractor:

    def __init__(self):
        self.modality_patterns = {
            ModalityType.PROHIBITION: [
                r"\bshall not\b",
                r"\bmust not\b",
                r"\bis prohibited from\b",
                r"\bis forbidden to\b",
                r"\bmay not\b",
            ],
            ModalityType.OBLIGATION: [
                r"\bshall\b",
                r"\bmust\b",
                r"\bis required to\b",
                r"\bis obligated to\b",
                r"\bhas to\b",
            ],
            ModalityType.PERMISSION: [
                r"\bmay\b",
                r"\bis permitted to\b",
                r"\bis allowed to\b",
                r"\bhas permission to\b",
            ],
            ModalityType.RECOMMENDATION: [
                r"\bshould\b",
                r"\bought to\b",
                r"\bis recommended to\b",
                r"\bis advised to\b",
            ],
            ModalityType.CAPABILITY: [
                r"\bcan\b",
                r"\bis able to\b",
                r"\bis capable of\b",
                r"\bhas the ability to\b",
            ],
        }

        self.condition_patterns = [
            r"(?:if|when|whenever|in case|in the event that)\s+([^,;.]+?)(?:[,;.]|$)",
            r"(?:provided that|on condition that)\s+([^,;.]+?)(?:[,;.]|$)",
            r"(?:where|wherever)\s+([^,;.]+?)(?:[,;.]|$)",
        ]

        self.temporal_patterns = [
            r"\b(?:immediately|promptly|instantly)\b",
            r"\b(?:within|in)\s+\d+\s+(?:seconds?|minutes?|hours?|days?)\b",
            r"\b(?:before|after|by|no later than)\s+([^,;.]+?)(?:[,;.]|$)",
            r"\b(?:as soon as possible|ASAP)\b",
            r"\b(?:at|on)\s+([^,;.]+?)(?:[,;.]|$)",
        ]

        self.negation_patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bnone\b",
            r"\bwithout\b",
        ]

        self.subject_patterns = [
            r"\b(?:the|a|an)\s+([A-Z][a-z]+(?:\s+[a-z]+)*?)\s+(?:shall|must|may|should|can|is|are|has|have)",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:shall|must|may|should|can|is|are|has|have)",
            r"\b(?:users?|operators?|administrators?|system|component|module|application|service)\b",
        ]

        self.object_patterns = [
            r"(?:shall|must|may|should|can|is|are|has|have)\s+(?:to\s+)?(?:be\s+)?([a-z]+(?:\s+[a-z]+)*?)(?:[,;.]|$)",
            r"(?:process|handle|manage|store|retrieve|validate|verify|authenticate|authorize)\s+([a-z]+(?:\s+[a-z]+)*?)(?:[,;.]|$)",
        ]

        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Any]:
        compiled: Dict[str, Any] = {}

        compiled["modality"]: Dict[ModalityType, List[Any]] = {}
        for modality_type, patterns in self.modality_patterns.items():
            compiled["modality"][modality_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        compiled["condition"] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.condition_patterns
        ]
        compiled["temporal"] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.temporal_patterns
        ]
        compiled["negation"] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.negation_patterns
        ]
        compiled["subject"] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.subject_patterns
        ]
        compiled["object"] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.object_patterns
        ]

        return compiled

    def extract_semantic_profile(self, sentence: str) -> SemanticProfile:
        profile = SemanticProfile(original_sentence=sentence)

        profile.condition = self._extract_condition(sentence)
        profile.subject = self._extract_subject(sentence)
        profile.modality = self._extract_modality(sentence)
        profile.object = self._extract_object(sentence)
        profile.temporal = self._extract_temporal(sentence)
        profile.negation = self._extract_negation(sentence)

        profile.confidence = self._calculate_profile_confidence(profile)

        return profile

    def extract_profiles_from_segment(
        self, segment: DocumentSegment
    ) -> List[SemanticProfile]:
        profiles = []
        sentences = self._split_into_sentences(segment.text)

        for sentence in sentences:
            if sentence.strip():  # Skip empty sentences
                profile = self.extract_semantic_profile(sentence)
                profile.metadata["segment_id"] = id(segment)
                profile.metadata["segment_type"] = segment.element_type.value
                profiles.append(profile)

        return profiles

    def _extract_modality(self, text: str) -> Optional[SemanticAnchor]:
        matches = []

        for modality_type, patterns in self.compiled_patterns["modality"].items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    matches.append(
                        {
                            "type": modality_type,
                            "text": match.group(0),
                            "start": match.start(),
                            "end": match.end(),
                            "pattern": pattern.pattern,
                        }
                    )

        if not matches:
            return None

        matches.sort(key=lambda x: x["start"])

        obligation_matches = [
            m for m in matches if m["type"] == ModalityType.OBLIGATION
        ]
        prohibition_matches = [
            m for m in matches if m["type"] == ModalityType.PROHIBITION
        ]

        if obligation_matches and prohibition_matches:
            first_match = matches[0]
            return SemanticAnchor(
                anchor_type="modality",
                text=first_match["text"],
                start_pos=first_match["start"],
                end_pos=first_match["end"],
                confidence=0.9,
            )

        first_match = matches[0]
        return SemanticAnchor(
            anchor_type="modality",
            text=first_match["text"],
            start_pos=first_match["start"],
            end_pos=first_match["end"],
            confidence=0.9,
        )

    def _extract_condition(self, text: str) -> Optional[SemanticAnchor]:
        for pattern in self.compiled_patterns["condition"]:
            match = pattern.search(text)
            if match:
                condition_text = match.group(1) if match.groups() else match.group(0)
                return SemanticAnchor(
                    anchor_type="condition",
                    text=condition_text.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                )
        return None

    def _extract_temporal(self, text: str) -> Optional[SemanticAnchor]:
        for pattern in self.compiled_patterns["temporal"]:
            match = pattern.search(text)
            if match:
                temporal_text = match.group(1) if match.groups() else match.group(0)
                return SemanticAnchor(
                    anchor_type="temporal",
                    text=temporal_text.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7,
                )
        return None

    def _extract_negation(self, text: str) -> Optional[SemanticAnchor]:
        for pattern in self.compiled_patterns["negation"]:
            match = pattern.search(text)
            if match:
                return SemanticAnchor(
                    anchor_type="negation",
                    text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                )
        return None

    def _extract_subject(self, text: str) -> Optional[SemanticAnchor]:
        for pattern in self.compiled_patterns["subject"]:
            match = pattern.search(text)
            if match:
                subject_text = match.group(1) if match.groups() else match.group(0)
                return SemanticAnchor(
                    anchor_type="subject",
                    text=subject_text.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.6,
                )
        return None

    def _extract_object(self, text: str) -> Optional[SemanticAnchor]:
        for pattern in self.compiled_patterns["object"]:
            match = pattern.search(text)
            if match:
                object_text = match.group(1) if match.groups() else match.group(0)
                return SemanticAnchor(
                    anchor_type="object",
                    text=object_text.strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.5,
                )
        return None

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_profile_confidence(self, profile: SemanticProfile) -> float:
        anchors = [
            profile.condition,
            profile.subject,
            profile.modality,
            profile.object,
            profile.temporal,
            profile.negation,
        ]

        valid_anchors = [a for a in anchors if a is not None]
        if not valid_anchors:
            return 0.0

        return sum(a.confidence for a in valid_anchors) / len(valid_anchors)

    def get_profiles_by_modality(
        self, profiles: List[SemanticProfile], modality_type: ModalityType
    ) -> List[SemanticProfile]:
        filtered = []
        for profile in profiles:
            if profile.modality:
                for pattern in self.modality_patterns.get(modality_type, []):
                    if re.search(pattern, profile.modality.text, re.IGNORECASE):
                        filtered.append(profile)
                        break
        return filtered

    def find_profiles_with_conditions(
        self, profiles: List[SemanticProfile]
    ) -> List[SemanticProfile]:
        return [p for p in profiles if p.condition is not None]

    def find_profiles_with_temporal_constraints(
        self, profiles: List[SemanticProfile]
    ) -> List[SemanticProfile]:
        return [p for p in profiles if p.temporal is not None]
