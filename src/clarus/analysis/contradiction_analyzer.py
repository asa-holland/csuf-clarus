from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import sys
import logging

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
)
logger = logging.getLogger("clarus.contradiction_analyzer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(_handler)
logger.propagate = False


class ContradictionType(Enum):
    MODALITY = "modality"  # Different modalities (must vs. must not)
    TEMPORAL = "temporal"  # Conflicting time constraints
    CONDITIONAL = "conditional"  # Contradictory conditions
    SEMANTIC = "semantic"  # General semantic contradiction
    TERMINOLOGY = "terminology"  # Inconsistent term usage
    LOGICAL = "logical"  # Logical contradictions


@dataclass
class SemanticProfile:

    subject: str
    predicate: str
    obj: str
    modality: str  # "must", "should", "may", etc.
    negation: bool
    condition: Optional[str] = None
    numerical_values: List[float] = field(default_factory=list)
    source_text: str = ""
    span: Tuple[int, int] = (0, 0)  # Character offsets in original text


@dataclass
class Contradiction:

    contradiction_type: ContradictionType
    statement_a: str
    statement_b: str
    confidence: float
    explanation: str
    evidence: Dict[str, str] = field(default_factory=dict)
    semantic_profile_a: Optional[SemanticProfile] = None
    semantic_profile_b: Optional[SemanticProfile] = None
    metadata: Dict = field(default_factory=dict)


class ContradictionDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            "min_confidence": 0.7,
            "enable_ml": False,
            "ml_threshold": 0.8,
            "use_candidate_filter": True,
            # Maximum distance (in segment indices) between two statements that
            # can be flagged as a contradiction.  Statements farther apart are
            # likely from different sections/topics and cross-comparisons produce
            # noise.  Set to None to disable the limit.
            "max_segment_distance": 10,
            **({} if config is None else config),
        }
        self.initialized = False
        self.ml_model = None
        self.embedding_model = None
        self.term_variants = {}  # For terminology consistency

    def initialize(self):
        if not self.initialized:
            if self.config["enable_ml"]:
                self._initialize_ml_components()
            self.initialized = True

    def _initialize_ml_components(self):
        # TODO: implement ML/NLI model
        try:
            pass
        except ImportError:
            print(
                "Warning: ML components not available. Falling back to rule-based only."
            )
            self.config["enable_ml"] = False

    def detect_contradictions(
        self, statements: List[str], context: Optional[Dict] = None
    ) -> List[Contradiction]:
        self.initialize()
        contradictions = []

        logger.info("detect_contradictions: received %d statement(s)", len(statements))
        for idx, s in enumerate(statements):
            logger.debug("  [%d] %s", idx, s)

        if not statements or len(statements) < 2:
            logger.info("detect_contradictions: fewer than 2 statements — skipping")
            return contradictions

        candidate_pairs = self._generate_candidate_pairs(statements, context)
        logger.info(
            "detect_contradictions: %d candidate pair(s) generated",
            len(candidate_pairs),
        )

        for i, j in candidate_pairs:
            if i >= len(statements) or j >= len(statements):
                continue

            profile_a = self._create_semantic_profile(statements[i])
            profile_b = self._create_semantic_profile(statements[j])

            detected = self._check_contradiction_pair(
                statements[i], statements[j], profile_a, profile_b, context
            )

            if detected:
                contradictions.append(detected)

        return sorted(contradictions, key=lambda x: x.confidence, reverse=True)

    def _generate_candidate_pairs(
        self, statements: List[str], context: Optional[Dict] = None
    ) -> List[Tuple[int, int]]:
        if not self.config["use_candidate_filter"]:
            return [
                (i, j)
                for i in range(len(statements))
                for j in range(i + 1, len(statements))
            ]

        max_dist = self.config.get("max_segment_distance")
        candidates = []
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                if max_dist is not None and (j - i) > max_dist:
                    continue
                if self._share_significant_terms(statements[i], statements[j]):
                    candidates.append((i, j))
        return candidates

    def _share_significant_terms(
        self, text_a: str, text_b: str, min_overlap: int = 1
    ) -> bool:
        # TODO: look at TF-IDF or embeddings
        words_a = set(re.findall(r"\b\w+\b", text_a.lower()))
        words_b = set(re.findall(r"\b\w+\b", text_b.lower()))
        common = words_a.intersection(words_b)

        significant = {
            w for w in common if len(w) > 3 and w not in self._get_stopwords()
        }
        return len(significant) >= min_overlap

    def _create_semantic_profile(self, text: str) -> SemanticProfile:
        # TODO: convert to "real NLP"
        return SemanticProfile(
            subject=self._extract_subject(text),
            predicate=self._extract_predicate(text),
            obj=self._extract_object(text),
            modality=self._extract_modality(text),
            negation="not" in text.lower().split(),
            source_text=text,
            span=(0, len(text)),
        )

    def _check_contradiction_pair(
        self,
        text_a: str,
        text_b: str,
        profile_a: SemanticProfile,
        profile_b: SemanticProfile,
        context: Optional[Dict] = None,
    ) -> Optional[Contradiction]:
        direct_contra = self._check_direct_contradiction(text_a, text_b)
        if direct_contra:
            return Contradiction(
                contradiction_type=ContradictionType.MODALITY,
                statement_a=text_a,
                statement_b=text_b,
                confidence=0.9,
                explanation="Direct contradiction detected between statements",
                evidence={"type": "direct_contradiction", "rule": "modality_mismatch"},
                semantic_profile_a=profile_a,
                semantic_profile_b=profile_b,
            )

        if self.config["enable_ml"]:
            ml_result = self._check_ml_contradiction(text_a, text_b)
            if ml_result and ml_result["confidence"] > self.config["ml_threshold"]:
                return Contradiction(
                    contradiction_type=ContradictionType.SEMANTIC,
                    statement_a=text_a,
                    statement_b=text_b,
                    confidence=ml_result["confidence"],
                    explanation=ml_result.get(
                        "explanation", "Potential contradiction detected"
                    ),
                    evidence={
                        "type": "ml_detected",
                        "model_confidence": ml_result["confidence"],
                    },
                    semantic_profile_a=profile_a,
                    semantic_profile_b=profile_b,
                )

        return None

    def _extract_subject(self, text: str) -> str:
        """Extract subject from text (simplified)"""
        # TODO, this would use dependency parsing
        return text.split()[0] if text else ""

    def _extract_predicate(self, text: str) -> str:
        """Extract predicate from text (simplified)"""
        # TODO, this would use dependency parsing
        words = text.split()
        return " ".join(words[1:-1]) if len(words) > 2 else ""

    def _extract_object(self, text: str) -> str:
        """Extract object from text (simplified)"""
        # TODO, this would use dependency parsing
        words = text.split()
        return words[-1] if words else ""

    def _extract_modality(self, text: str) -> str:
        """Extract modality from text (simplified)"""
        modals = ["must", "shall", "should", "may", "can", "will"]
        words = text.lower().split()
        for word in words:
            if word in modals:
                return word
        return ""

    def _check_direct_contradiction(self, text_a: str, text_b: str) -> bool:
        text_a = text_a.lower()
        text_b = text_b.lower()

        # Negated-verb contradictions: one text has "not/cannot/never VERB", the other has
        # the positive form.  Covers patterns like "do not learn" vs "learn", "cannot die"
        # vs "they die", etc. that are common in game-rules and natural-language specs.
        negatable_verbs = [
            "learn", "die", "vote", "win", "lose", "execute", "nominate",
            "target", "protect", "choose", "use",
        ]
        neg_re_str = r"(?:not|cannot|can't|does\s+not|do\s+not|never)\s+{v}"
        for verb in negatable_verbs:
            neg_re = re.compile(neg_re_str.format(v=verb))
            pos_re = re.compile(rf"\b{verb}\b")
            a_neg = bool(neg_re.search(text_a))
            b_neg = bool(neg_re.search(text_b))
            a_pos = bool(pos_re.search(text_a)) and not a_neg
            b_pos = bool(pos_re.search(text_b)) and not b_neg
            if (a_neg and b_pos) or (b_neg and a_pos):
                return True

        # Game-outcome antonyms (e.g. "good wins" vs "evil wins", "good loses" vs "evil wins")
        outcome_pairs = [
            (r"\bgood\s+wins\b", r"\bevil\s+wins\b"),
            (r"\bgood\s+loses\b", r"\bevil\s+wins\b"),
            (r"\bgood\s+wins\b", r"\bgood\s+loses\b"),
        ]
        for pat_pos, pat_neg in outcome_pairs:
            if (re.search(pat_pos, text_a) and re.search(pat_neg, text_b)) or (
                re.search(pat_pos, text_b) and re.search(pat_neg, text_a)
            ):
                return True

        modal_contradictions = [
            (r"\bmust\b", r"must not"),
            (r"\bshall\b", r"shall not"),
            (r"\bshould\b", r"should not"),
            (r"\brequired to\b", r"not required to"),
            (r"\bprohibited from\b", r"allowed to"),
            (r"\bnever\b", r"always"),
            (r"\bnone\b", r"some"),
            (r"\bnothing\b", r"something"),
        ]

        for pos, neg in modal_contradictions:
            if (re.search(pos, text_a) and re.search(neg, text_b)) or (
                re.search(pos, text_b) and re.search(neg, text_a)
            ):
                return True

        value_antonyms = [
            ("true", "false"),
            ("yes", "no"),
            ("enabled", "disabled"),
            ("active", "inactive"),
            ("valid", "invalid"),
            ("allowed", "prohibited"),
            ("required", "optional"),
            ("present", "absent"),
            ("correct", "incorrect"),
            ("on", "off"),
            ("supported", "unsupported"),
            ("mandatory", "optional"),
            ("positive", "negative"),
            ("high", "low"),
            ("greater than", "less than"),
            ("more than", "fewer than"),
            ("above", "below"),
        ]
        for val_a, val_b in value_antonyms:
            pat_a = rf"\bis\s+{val_a}\b"
            pat_b = rf"\bis\s+{val_b}\b"
            if (re.search(pat_a, text_a) and re.search(pat_b, text_b)) or (
                re.search(pat_a, text_b) and re.search(pat_b, text_a)
            ):
                if self._share_significant_terms(text_a, text_b):
                    logger.debug(
                        "Value antonym contradiction: '%s' vs '%s' (pair: %s/%s)",
                        text_a[:60],
                        text_b[:60],
                        val_a,
                        val_b,
                    )
                    return True

        num_a = set(re.findall(r"\b\d+(?:\.\d+)?\b", text_a))
        num_b = set(re.findall(r"\b\d+(?:\.\d+)?\b", text_b))

        if num_a and num_b:
            quantity_terms = [
                "at least",
                "at most",
                "less than",
                "more than",
                "greater than",
                "fewer than",
            ]
            has_quantity_a = any(term in text_a for term in quantity_terms)
            has_quantity_b = any(term in text_b for term in quantity_terms)

            if has_quantity_a or has_quantity_b:
                if num_a != num_b:
                    return True

        return False

    def _check_ml_contradiction(self, text_a: str, text_b: str) -> Optional[Dict]:
        # TODO: implement ML/NLI model
        if not self.config["enable_ml"] or self.ml_model is None:
            return None

        try:

            return {
                "contradiction": True,
                "confidence": 0.85,
                "explanation": "ML model detected potential contradiction",
            }
        except Exception as e:
            print(f"Error in ML contradiction check: {e}")
            return None

    def _get_stopwords(self) -> Set[str]:
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "at",
            "from",
            "by",
            "on",
            "off",
            "for",
            "in",
            "out",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "couldn",
            "didn",
            "doesn",
            "hadn",
            "hasn",
            "haven",
            "isn",
            "ma",
            "mightn",
            "mustn",
            "needn",
            "shan",
            "shouldn",
            "wasn",
            "weren",
            "won",
            "wouldn",
        }


if __name__ == "__main__":
    detector = ContradictionDetector(config={"enable_ml": False})

    statements = [
        "Users must authenticate with a password.",
        "Users must not authenticate with a password.",
        "The system requires authentication via biometrics.",
        "Authentication is optional for guest users.",
    ]

    contradictions = detector.detect_contradictions(statements)

    print(f"Found {len(contradictions)} potential contradictions:")
    for i, contra in enumerate(contradictions, 1):
        print(f"\nContradiction {i}:")
        print(f"  Type: {contra.contradiction_type.value}")
        print(f"  Statement 1: {contra.statement_a}")
        print(f"  Statement 2: {contra.statement_b}")
        print(f"  Confidence: {contra.confidence:.2f}")
        print(f"  Explanation: {contra.explanation}")
