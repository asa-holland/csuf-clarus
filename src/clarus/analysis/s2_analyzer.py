from __future__ import annotations

import re
import sys
import logging
from dataclasses import dataclass
from typing import List, Set, Tuple

from .document_processor import DocumentSegment, ElementType

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
)
logger = logging.getLogger("clarus.s2_analyzer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(_handler)
logger.propagate = False


@dataclass
class S2Finding:
    uid: str
    error_type: str
    category: str
    severity: str
    text_span: str
    context: str
    explanation: str
    segment_type: str
    confidence: float


_OBLIGATION = re.compile(
    r"\b(shall|must|is required to|are required to|has to|have to"
    r"|is mandatory|are mandatory|requires?|mandate[sd]?)\b",
    re.IGNORECASE,
)
_PROHIBITION = re.compile(
    r"\b(shall not|must not|may not|is prohibited|is forbidden"
    r"|is not permitted|is not allowed|cannot|can not)\b",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"\b(may|can|is permitted to|are permitted to"
    r"|is allowed to|are allowed to|has permission to)\b",
    re.IGNORECASE,
)

_ROLE_BEFORE_MODAL = re.compile(
    r"\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
    r"(?:shall|must|should|may|can|will)\b",
)

_HIDDEN_NORMATIVE = [
    re.compile(
        r"\b(?:should|ought to)\s+(?:always|never|only|not|be)\b", re.IGNORECASE
    ),
    re.compile(r"\bone\s+should\b", re.IGNORECASE),
    re.compile(r"\bwe\s+should\b", re.IGNORECASE),
    re.compile(r"\ball\s+\w+\s+should\b", re.IGNORECASE),
    re.compile(
        r"\bit\s+is\s+(?:expected|recommended|required|necessary|mandatory)\s+that\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bis\s+expected\s+to\b", re.IGNORECASE),
    re.compile(r"\bmust\s+(?:always|never|only)\b", re.IGNORECASE),
]

_VAGUE_QUALIFIER = [
    re.compile(r"\btoo\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bvery\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bquite\s+\w+\b", re.IGNORECASE),
    re.compile(r"\brather\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bsomewhat\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bapproximately\b", re.IGNORECASE),
    re.compile(r"\bapprox\.?\b", re.IGNORECASE),
    re.compile(r"\babout\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bgenerally\b", re.IGNORECASE),
    re.compile(r"\btypically\b", re.IGNORECASE),
    re.compile(r"\busually\b", re.IGNORECASE),
    re.compile(r"\bnormally\b", re.IGNORECASE),
    re.compile(r"\broughly\b", re.IGNORECASE),
    re.compile(r"\bsufficient(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\breasonable(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bappropriate(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\badequate(?:ly)?\b", re.IGNORECASE),
]

_AMBIGUOUS_PRONOUN = re.compile(
    r"\b(it|this|that|they|them|its|their|these|those)\s+"
    r"(?:shall|must|should|may|can|will|would|could|might|is|are|was|were)\b",
    re.IGNORECASE,
)

_VAGUE_TEMPORAL = [
    re.compile(r"\bas\s+soon\s+as\s+possible\b", re.IGNORECASE),
    re.compile(r"\bASAP\b"),
    re.compile(r"\bpromptly\b", re.IGNORECASE),
    re.compile(r"\bquickly\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+(?:undue\s+)?delay\b", re.IGNORECASE),
    re.compile(r"\bin\s+a\s+timely\s+(?:fashion|manner|way)\b", re.IGNORECASE),
    re.compile(r"\bsoon\b", re.IGNORECASE),
    re.compile(r"\beventually\b", re.IGNORECASE),
    re.compile(r"\bin\s+due\s+course\b", re.IGNORECASE),
    re.compile(
        r"\bwhen\s+(?:convenient|possible|feasible|appropriate)\b", re.IGNORECASE
    ),
    re.compile(r"\bat\s+the\s+earliest\s+(?:opportunity|convenience)\b", re.IGNORECASE),
]

_TIME_SENSITIVE_VERBS = re.compile(
    r"\b(notify|notif(?:y|ies|ied|ication)|report|submit|deliver|respond|forward|transmit"
    r"|complete|provide|file|disclose|update|renew|confirm|acknowledge|escalate|alert"
    r"|return|remit|deposit|produce|supply|furnish)\b",
    re.IGNORECASE,
)
_SPECIFIC_TEMPORAL = re.compile(
    r"\bwithin\s+\d+\b"
    r"|\bno\s+later\s+than\b"
    r"|\bby\s+(?:the\s+)?(?:\d|end|close|start|beginning)\b"
    r"|\bbefore\s+\w"
    r"|\bafter\s+\w"
    r"|\d+\s+(?:second|minute|hour|day|week|month|year)s?\b"
    r"|\b(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\b",
    re.IGNORECASE,
)

_ROLE_SYNONYM_GROUPS: List[Set[str]] = [
    {
        "operator",
        "technician",
        "worker",
        "employee",
        "personnel",
        "staff",
        "practitioner",
    },
    {"administrator", "admin", "superuser", "manager", "supervisor"},
    {"user", "end-user", "client", "customer", "subscriber", "member", "patron"},
    {"contractor", "vendor", "supplier", "provider"},
    {"applicant", "submitter", "requester", "requestor", "petitioner"},
    {"authority", "regulator", "agency", "department", "bureau", "office"},
    {"recipient", "receiver", "addressee", "beneficiary"},
    {"owner", "holder", "licensee", "registrant"},
]

_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
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
    "here",
    "there",
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
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "with",
    "that",
    "this",
    "it",
    "its",
    "also",
    "been",
    "being",
}


def _keywords(text: str) -> Set[str]:
    return {
        w.lower()
        for w in re.findall(r"\b\w+\b", text)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }


class S2Analyzer:
    def analyze(self, text: str, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        findings.extend(self._check_fail002(segments))
        findings.extend(self._check_fail004(segments))
        findings.extend(self._check_fail005(segments))
        findings.extend(self._check_fail006(segments))
        findings.extend(self._check_fail007(segments))
        findings.extend(self._check_fail008(segments))
        by_uid: dict = {}
        for f in findings:
            by_uid[f.uid] = by_uid.get(f.uid, 0) + 1
        logger.info("S2Analyzer: %d finding(s) — %s", len(findings), by_uid)
        return findings

    def _check_fail002(self, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        norm_segs = [s for s in segments if s.element_type == ElementType.NORMATIVE]

        role_map: dict = {}  # lower_form → (original_form, context_snippet)
        for seg in norm_segs:
            for match in _ROLE_BEFORE_MODAL.finditer(seg.text):
                role = match.group(1).strip()
                key = role.lower()
                if key not in role_map:
                    role_map[key] = (role, seg.text[:200])

        if len(role_map) < 2:
            return findings

        roles = list(role_map.items())
        flagged: Set[Tuple[str, str]] = set()

        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                a_key, (a_orig, a_ctx) = roles[i]
                b_key, (b_orig, b_ctx) = roles[j]

                if a_key == b_key or a_key in b_key or b_key in a_key:
                    continue

                pair = tuple(sorted([a_key, b_key]))
                if pair in flagged:
                    continue

                for group in _ROLE_SYNONYM_GROUPS:
                    if a_key in group and b_key in group:
                        flagged.add(pair)
                        findings.append(
                            S2Finding(
                                uid="FAIL-002",
                                error_type="Synonym Inconsistency",
                                category="Terminology",
                                severity="Medium",
                                text_span=f'"{a_orig}" vs "{b_orig}"',
                                context=(
                                    f"First occurrence: {a_ctx[:150]}"
                                    f" | Second occurrence: {b_ctx[:150]}"
                                ),
                                explanation=(
                                    f'Role terms "{a_orig}" and "{b_orig}" appear to refer to '
                                    f"the same actor class but use different names. If they "
                                    f"denote the same role, the inconsistency creates ambiguity "
                                    f"about who bears each obligation."
                                ),
                                segment_type="normative",
                                confidence=0.75,
                            )
                        )
                        break

        return findings

    def _check_fail004(self, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        norm_segs = [s for s in segments if s.element_type == ElementType.NORMATIVE]

        ob_segs = [
            (s, _keywords(s.text))
            for s in norm_segs
            if _OBLIGATION.search(s.text) and not _PROHIBITION.search(s.text)
        ]
        per_segs = [
            (s, _keywords(s.text))
            for s in norm_segs
            if _PERMISSION.search(s.text) and not _OBLIGATION.search(s.text)
        ]

        flagged: Set[Tuple[int, int]] = set()

        for ob_seg, ob_kw in ob_segs:
            for per_seg, per_kw in per_segs:
                pair = (id(ob_seg), id(per_seg))
                if pair in flagged:
                    continue
                shared = ob_kw & per_kw
                if len(shared) >= 2:
                    flagged.add(pair)
                    ob_m = _OBLIGATION.search(ob_seg.text)
                    per_m = _PERMISSION.search(per_seg.text)
                    findings.append(
                        S2Finding(
                            uid="FAIL-004",
                            error_type="Modal Inconsistency",
                            category="Logical",
                            severity="Medium",
                            text_span=(
                                f'"{ob_m.group(0) if ob_m else "obligation"}"'
                                f' vs "{per_m.group(0) if per_m else "permission"}"'
                            ),
                            context=(
                                f"Obligation: {ob_seg.text[:200]}"
                                f" | Permission: {per_seg.text[:200]}"
                            ),
                            explanation=(
                                f'Shared topic ({", ".join(sorted(shared)[:4])}) is governed '
                                f"by an obligation modal in one statement and a permission modal "
                                f"in another, potentially creating a compliance loophole."
                            ),
                            segment_type="normative",
                            confidence=0.70,
                        )
                    )

        return findings

    def _check_fail005(self, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        for seg in segments:
            if seg.element_type != ElementType.INFORMATIVE:
                continue
            for pattern in _HIDDEN_NORMATIVE:
                match = pattern.search(seg.text)
                if match:
                    findings.append(
                        S2Finding(
                            uid="FAIL-005",
                            error_type="Hidden Normative",
                            category="Context",
                            severity="Medium",
                            text_span=match.group(0),
                            context=seg.text[:300],
                            explanation=(
                                f'Normative-strength language ("{match.group(0)}") appears in a '
                                f"segment classified as informative. Requirements embedded in "
                                f"informative sections may be missed during compliance review."
                            ),
                            segment_type="informative",
                            confidence=0.80,
                        )
                    )
                    break

        return findings

    def _check_fail006(self, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        for seg in segments:
            if seg.element_type not in (ElementType.NORMATIVE, ElementType.UNKNOWN):
                continue
            if not _OBLIGATION.search(seg.text):
                continue
            for pattern in _VAGUE_QUALIFIER:
                match = pattern.search(seg.text)
                if match:
                    findings.append(
                        S2Finding(
                            uid="FAIL-006",
                            error_type="Vague Qualifier",
                            category="Vagueness",
                            severity="Low",
                            text_span=match.group(0),
                            context=seg.text[:300],
                            explanation=(
                                f'Vague qualifier "{match.group(0)}" in a normative statement '
                                f"creates a non-verifiable compliance threshold."
                            ),
                            segment_type=seg.element_type.value,
                            confidence=0.65,
                        )
                    )
                    break

        return findings

    def _check_fail007(self, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        for seg in segments:
            if seg.element_type not in (ElementType.NORMATIVE, ElementType.UNKNOWN):
                continue
            match = _AMBIGUOUS_PRONOUN.search(seg.text)
            if match:
                findings.append(
                    S2Finding(
                        uid="FAIL-007",
                        error_type="Ambiguous Referent",
                        category="Vagueness",
                        severity="Medium",
                        text_span=match.group(0),
                        context=seg.text[:300],
                        explanation=(
                            f'Pronoun "{match.group(1)}" in a normative statement has an '
                            f"ambiguous referent — the requirement subject cannot be determined "
                            f"without additional context."
                        ),
                        segment_type=seg.element_type.value,
                        confidence=0.70,
                    )
                )

        return findings

    def _check_fail008(self, segments: List[DocumentSegment]) -> List[S2Finding]:
        findings: List[S2Finding] = []
        for seg in segments:
            if seg.element_type not in (ElementType.NORMATIVE, ElementType.UNKNOWN):
                continue
            if not _OBLIGATION.search(seg.text):
                continue

            for pattern in _VAGUE_TEMPORAL:
                match = pattern.search(seg.text)
                if match:
                    findings.append(
                        S2Finding(
                            uid="FAIL-008",
                            error_type="Missing Temporal Anchor",
                            category="Logical",
                            severity="High",
                            text_span=match.group(0),
                            context=seg.text[:300],
                            explanation=(
                                f'Vague temporal expression "{match.group(0)}" provides no '
                                f"measurable compliance deadline."
                            ),
                            segment_type=seg.element_type.value,
                            confidence=0.85,
                        )
                    )
                    break

            else:
                verb_match = _TIME_SENSITIVE_VERBS.search(seg.text)
                if verb_match and not _SPECIFIC_TEMPORAL.search(seg.text):
                    findings.append(
                        S2Finding(
                            uid="FAIL-008",
                            error_type="Missing Temporal Anchor",
                            category="Logical",
                            severity="High",
                            text_span=verb_match.group(0),
                            context=seg.text[:300],
                            explanation=(
                                f'Time-sensitive verb "{verb_match.group(0)}" in a normative '
                                f"statement with no specific deadline or temporal qualifier "
                                f'(e.g., "within N days", "no later than").'
                            ),
                            segment_type=seg.element_type.value,
                            confidence=0.70,
                        )
                    )

        return findings
