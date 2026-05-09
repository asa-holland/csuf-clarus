import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .document_processor import DocumentSegment


class ModalityType(Enum):

    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"
    CAPABILITY = "capability"
    NONE = "none"


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
            r"(?:provided that|on condition that|assuming that|given that)\s+([^,;.]+?)(?:[,;.]|$)",
            r"(?:as long as|so long as)\s+([^,;.]+?)(?:[,;.]|$)",
            r"(?:unless|except\s+(?:when|if))\s+([^,;.]+?)(?:[,;.]|$)",
            r"(?:where|wherever)\s+([^,;.]+?)(?:[,;.]|$)",
        ]

        self.temporal_patterns = [
            # Quantity + unit duration
            r"\b(?:within|in)\s+\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
            # Point-in-time adverbs
            r"\b(?:immediately|promptly|instantly|now|then|soon|already|finally|subsequently|previously|thereafter|eventually)\b",
            # Frequency adverbs
            r"\b(?:always|never|once|twice|repeatedly|daily|weekly|nightly)\b",
            # Temporal preposition + NP
            r"\b(?:before|after|by|until|since|during|throughout)\s+([^,;.]+?)(?:[,;.]|$)",
            # "no later than" / "as soon as possible"
            r"\b(?:no later than|as soon as possible|ASAP)\b",
            # each/every/per + any time word
            r"\b(?:each|every|per)\s+(\w+)\b",
            # once per [period]
            r"\bonce\s+per\s+\w+\b",
            # at/on + NP (keep broad — temporal prepositions before a noun phrase)
            r"\b(?:at|on)\s+([^,;.]+?)(?:[,;.]|$)",
            # the next/first/last/following + [period word]
            r"\b(?:the\s+)?(?:next|first|last|following)\s+\w+\b",
        ]

        self.negation_patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bnone\b",
            r"\bwithout\b",
            r"\bcannot\b",
            r"\bcan't\b",
            r"\bwon't\b",
            r"\bisn't\b",
            r"\baren't\b",
            r"\bdoesn't\b",
            r"\bdon't\b",
        ]

        self.subject_patterns = [
            # Personal/impersonal pronouns
            r"\b(I|you|he|she|it|we|they|one)\b",
            # Quantifier + (optional modifiers) + noun, followed by any auxiliary
            r"\b((?:each|every|all|any|some|no|another|both|either|neither)\s+"
            r"(?:[a-z]+\s+){0,2}[a-z]+)\s+"
            r"(?:is|are|was|were|will|shall|must|may|should|can|could|would|might"
            r"|has|have|had|does|do|did|cannot|can't)\b",
            # Article + noun phrase, followed by any auxiliary
            r"\b(?:the|a|an)\s+((?:[a-z]+\s+){0,2}[a-z]+)\s+"
            r"(?:is|are|was|were|will|shall|must|may|should|can|could|would|might"
            r"|has|have|had|does|do|did|cannot|can't)\b",
            # Proper noun(s) before auxiliary/modal
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+"
            r"(?:is|are|was|were|will|shall|must|may|should|can|could|would|might)\b",
        ]

        self.object_patterns = [
            # After a modal/auxiliary: "must/may/shall/etc. [not] [to/be] <object>"
            r"(?:shall|must|may|should|can|will|would|could|might)\s+"
            r"(?:not\s+)?(?:to\s+)?(?:be\s+)?([a-z]+(?:\s+[a-z]+){0,4}?)(?:[,;.]|$)",
            # Common high-frequency transitive verbs + article/determiner + NP
            r"\b(?:choose|select|take|give|make|find|get|send|keep|hold|use|add|"
            r"change|show|move|bring|call|set|ask|put|turn|leave|pass|play|stop|"
            r"allow|require|include|affect|create|remove|replace|receive|provide|"
            r"reveal|announce|inform|tell|know|learn|discover)\s+"
            r"(?:a|an|the|another|two|three|no)?\s*([a-z]+(?:\s+[a-z]+){0,3}?)(?:[,;.:]|$)",
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
