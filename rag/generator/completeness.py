"""Bounded question obligations and deterministic coverage relationships, not semantic proof."""

import ast
import re

from api.schemas.questions import SupportDecision

_SPLIT = re.compile(r"\s+and\s+|;|\?\s+(?=\w)", re.I)
_QUOTED = re.compile(r"`[^`]*`|\"[^\"]*\"|(?<!\w)'[^']*'(?!\w)")
_NUMERIC = re.compile(
    r"\b(?:how many|how much|numeric|number|limit|bound|bytes?|seconds?|duration|timeout|count)\b",
    re.I,
)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")


def requirements(question):
    """Server-owned exact question spans; no model may delete or rename requirements.

    Connectors remain in the spans. The full original question is always authoritative.
    Quoted conjunctions are data, not additional obligations. Unsupported excess fails closed.
    """
    protected = [(m.start(), m.end()) for m in _QUOTED.finditer(question)]
    cuts = (
        [0]
        + [
            m.start()
            for m in _SPLIT.finditer(question)
            if not any(a <= m.start() < b for a, b in protected)
        ]
        + [len(question)]
    )
    units = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        text = question[start:end].strip(" ;?\n\t")
        if text:
            units.append({"requirement_id": "R" + str(len(units) + 1), "text": text})
    if not units or len(units) > 6:
        raise ValueError("Question requirements exceed bound")
    return units


def needs_number(unit, selected):
    if _NUMERIC.search(unit.replace("_", " ")):
        return True
    # An explicitly asked-about numeric source constant also requires its value.
    # Parse only already bounded canonical chunks; never evaluate or import source.
    for item in selected.values():
        if not item.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(item.excerpt)
        except (SyntaxError, ValueError, RecursionError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if type(node.value.value) not in (int, float):
                    continue
                if any(
                    isinstance(t, ast.Name) and re.search(r"\b" + re.escape(t.id) + r"\b", unit)
                    for t in node.targets
                ):
                    return True
    return False


def answer_span(text):
    # Cosmetic delimiter style only for balanced simple quoted literals. Contents and
    # every truth-changing word remain intact; canonical SOURCE quotes are never changed.
    text = re.sub(r"([\"'])([^\"'\n]+)\1", lambda m: '"' + m[2] + '"', text)
    return " ".join(text.split())


def validate_coverage(value, question, answer, selected):
    """Support AND all obligations AND exact claim/citation coverage are mandatory."""
    decision = SupportDecision.model_validate_json(value)
    units = requirements(question)
    expected = {u["requirement_id"] for u in units}
    actual = [c.requirement_id for c in decision.coverage]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete coverage")
    indices = [s.claim_index for s in decision.support]
    if len(indices) != len(set(indices)) or set(indices) != set(range(len(answer.claims))):
        raise ValueError("Unsupported output")
    for support in decision.support:
        claim = answer.claims[support.claim_index]
        refs = {r.evidence_id: r for r in claim.references}
        ids = support.evidence_ids
        if len(ids) != len(set(ids)) or set(ids) != set(refs):
            raise ValueError("Unsupported output")
        if not set(ids) <= set(selected) or not support.source_fact.strip():
            raise ValueError("Unsupported output")
        if support.verdict != "supported":
            raise ValueError(
                {
                    "contradicted": "Contradicted output",
                    "insufficient": "Insufficient support",
                    "invalid": "Unsupported output",
                }[support.verdict]
            )
    records = {c.requirement_id: c for c in decision.coverage}
    used_quotes = set()
    for unit in units:
        covered = records[unit["requirement_id"]]
        if not covered.covered or not covered.claim_indices or not covered.evidence_ids:
            raise ValueError("Incomplete coverage")
        if len(set(covered.claim_indices)) != len(covered.claim_indices):
            raise ValueError("Incomplete coverage")
        if any(i >= len(answer.claims) for i in covered.claim_indices):
            raise ValueError("Incomplete coverage")
        claims = [answer.claims[i] for i in covered.claim_indices]
        quote = answer_span(covered.answer_quote)
        if (
            not quote
            or quote not in " ".join(answer_span(c.text) for c in claims)
            or quote in used_quotes
        ):
            raise ValueError("Incomplete coverage")
        used_quotes.add(quote)
        cited = {ref.evidence_id for claim in claims for ref in claim.references}
        if not set(covered.evidence_ids) <= cited or not set(covered.evidence_ids) <= set(selected):
            raise ValueError("Incomplete coverage")
        if needs_number(unit["text"], selected):
            numbers = _NUMBER.findall(quote)
            # A symbolic placeholder alone is not the requested concrete value. Numeric
            # membership supplements the semantic per-unit judgment, never replaces it.
            backing = "\n".join(selected[key].excerpt for key in covered.evidence_ids)
            if not numbers or not any(n in _NUMBER.findall(backing) for n in numbers):
                raise ValueError("Incomplete coverage")
    return decision
