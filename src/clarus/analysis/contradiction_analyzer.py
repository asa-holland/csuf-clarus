from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import sys
import logging
import threading

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
)
logger = logging.getLogger("clarus.contradiction_analyzer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Module-level NLI CrossEncoder singleton.
# Shared across all ContradictionDetector instances to avoid reloading the
# model for every request.  Uses double-checked locking so only one thread
# ever triggers the (expensive) load.
#   _nli_model is None   → never attempted
#   _nli_model is False  → load failed; do not retry
#   _nli_model is <obj>  → ready to use
# ---------------------------------------------------------------------------
_nli_model_lock = threading.Lock()
_nli_model = None  # type: ignore[assignment]

# DeBERTa NLI label index mapping for cross-encoder/nli-deberta-v3-base
_NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def _get_nli_model():
    global _nli_model
    if _nli_model is not None:
        return _nli_model if _nli_model is not False else None
    with _nli_model_lock:
        if _nli_model is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                logger.info("Loading NLI model %s …", _NLI_MODEL_NAME)
                _nli_model = CrossEncoder(_NLI_MODEL_NAME)
                logger.info("NLI model loaded successfully")
            except Exception as exc:
                logger.warning(
                    "NLI model unavailable (%s) — rule-based detection only", exc
                )
                _nli_model = False
    return _nli_model if _nli_model is not False else None


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
            "enable_ml": True,
            "ml_threshold": 0.8,
            "use_candidate_filter": True,
            "max_segment_distance": 10,
            "min_term_overlap": 2,
            "min_nli_words": 10,
            **({} if config is None else config),
        }
        self.initialized = False
        self.ml_model = None
        self.embedding_model = None
        self.term_variants = {}

    def initialize(self):
        if not self.initialized:
            if self.config["enable_ml"]:
                self._initialize_ml_components()
            self.initialized = True

    def _initialize_ml_components(self):
        model = _get_nli_model()
        if model is not None:
            self.ml_model = model
        else:
            logger.warning(
                "ML components unavailable — disabling for this detector instance"
            )
            self.config["enable_ml"] = False

    def detect_contradictions(
        self, statements: List[str], context: Optional[Dict] = None
    ) -> List[Contradiction]:
        self.initialize()
        contradictions = []

        logger.info("detect_contradictions: received %d statement(s)", len(statements))

        statements = [s for s in statements if self._is_sentence_like(s)]
        logger.info(
            "detect_contradictions: %d statement(s) after sentence filter",
            len(statements),
        )
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

            if detected and detected.confidence >= self.config["min_confidence"]:
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
                if self._share_significant_terms(
                    statements[i],
                    statements[j],
                    min_overlap=self.config.get("min_term_overlap", 2),
                ):
                    candidates.append((i, j))
        return candidates

    # Verb tokens that signal a complete predicate.  A segment must contain at
    # least one of these to be treated as a sentence worth comparing.
    _VERB_SIGNALS: frozenset = frozenset({
        "must", "shall", "should", "may", "can", "will", "would", "could", "might",
        "is", "are", "was", "were", "has", "have", "had",
        "does", "do", "did",
        "require", "requires", "required",
        "provide", "provides", "provided",
        "allow", "allows", "allowed",
        "prohibit", "prohibits", "prohibited",
        "state", "states", "stated",
        "define", "defines", "defined",
        "specify", "specifies", "specified",
        "ensure", "ensures", "ensured",
        "contain", "contains", "contained",
        "include", "includes", "included",
        "apply", "applies", "applied",
    })

    def _is_sentence_like(self, text: str) -> bool:
        """Return True only for text that looks like a complete sentence.

        Rejects: headers, short labels, all-caps titles, colon-terminated
        fragments, and anything lacking a recognisable predicate verb.
        """
        text = text.strip()
        words = text.split()

        if len(words) < 8:
            return False

        # All-caps lines are almost always headings or acronym definitions
        alpha_only = re.sub(r"[^A-Za-z]", "", text)
        if alpha_only and alpha_only == alpha_only.upper():
            return False

        # "Label:" style fragments — ends with colon, no interior sentence punct
        if text.endswith(":") and "," not in text and "." not in text[:-1]:
            return False

        # Numbered or bulleted labels without a proper predicate
        # e.g. "1. Purpose" or "• Definitions"
        if re.match(r"^[\d]+[\.\)]\s+\S+$", text) or re.match(r"^[•\-\*]\s+\S+$", text):
            return False

        lower_words = set(re.findall(r"\b\w+\b", text.lower()))
        if not lower_words.intersection(self._VERB_SIGNALS):
            return False

        return True

    def _share_significant_terms(
        self, text_a: str, text_b: str, min_overlap: int = 1
    ) -> bool:
        clean_a = re.sub(r"\[.*?\]", " ", text_a).lower()
        clean_b = re.sub(r"\[.*?\]", " ", text_b).lower()
        words_a = set(re.findall(r"\b\w+\b", clean_a))
        words_b = set(re.findall(r"\b\w+\b", clean_b))
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
            if (
                ml_result
                and ml_result["is_conflict"]
                and ml_result["confidence"] > self.config["ml_threshold"]
            ):
                return Contradiction(
                    contradiction_type=ContradictionType.SEMANTIC,
                    statement_a=text_a,
                    statement_b=text_b,
                    confidence=ml_result["confidence"],
                    explanation=(
                        f"NLI model detected semantic contradiction "
                        f"(score: {ml_result['confidence']:.2f})"
                    ),
                    evidence={
                        "type": "ml_detected",
                        "model": _NLI_MODEL_NAME,
                        "label": ml_result["label"],
                        "model_confidence": str(round(ml_result["confidence"], 4)),
                    },
                    semantic_profile_a=profile_a,
                    semantic_profile_b=profile_b,
                )

        return None

    def _extract_subject(self, text: str) -> str:
        return text.split()[0] if text else ""

    def _extract_predicate(self, text: str) -> str:
        words = text.split()
        return " ".join(words[1:-1]) if len(words) > 2 else ""

    def _extract_object(self, text: str) -> str:
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

    _MIN_NLI_WORDS: int = 6

    def _check_ml_contradiction(self, text_a: str, text_b: str) -> Optional[Dict]:
        if not self.config["enable_ml"] or self.ml_model is None:
            return None

        min_words = self.config.get("min_nli_words", self._MIN_NLI_WORDS)
        words_a = len(text_a.split())
        words_b = len(text_b.split())
        if words_a < min_words or words_b < min_words:
            logger.debug(
                "NLI skipped — segments too short (%d / %d words, min %d)",
                words_a,
                words_b,
                min_words,
            )
            return None

        try:
            scores = self.ml_model.predict([(text_a, text_b)], apply_softmax=True)[0]
            contradiction_score = float(scores[0])
            top_idx = int(scores.argmax())
            label = _NLI_LABEL_MAP[top_idx]
            logger.debug(
                "NLI scores — contradiction=%.3f entailment=%.3f neutral=%.3f → %s",
                float(scores[0]),
                float(scores[1]),
                float(scores[2]),
                label,
            )
            return {
                "label": label,
                "confidence": contradiction_score,
                "is_conflict": label == "contradiction",
            }
        except Exception as exc:
            logger.warning("NLI inference failed: %s", exc)
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
    detector = ContradictionDetector()

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
