"""J2-only lexical queries and exact bounded windows; not semantic evidence validation."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from core.services import career_evidence
from rag.retriever.revision_keyword import tokenize

# Search scaffolding, not a technology/benchmark allowlist. Full obligations still
# reach the generator/checker unchanged; this does not decide evidence support.
_STOP = frozenset(
    "a an the to of from in on for with and or is are be by as at that this it "
    "provide include required preferred require requires should must using use "
    "function functions development implementation documentation documented "
    "experience ability able knowledge skills skill".split()
)


# Language scope comes from the explicit normalized skill/aliases, never a repo name.
_LANGUAGES = {
    "go": {".go"},
    "golang": {".go"},
    "python": {".py"},
    "py": {".py"},
    "javascript": {".js", ".jsx"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "rust": {".rs"},
    "ruby": {".rb"},
    "php": {".php"},
    "c": {".c", ".h"},
    "c++": {".cpp", ".cc", ".hpp", ".h"},
    "c#": {".cs"},
}
_CODE_SUFFIXES = frozenset(s for suffixes in _LANGUAGES.values() for s in suffixes)
_GENERIC_OBJECTS = {"value", "values", "document", "documents"}
_CALLABLE = re.compile(r"^\s*(?:async\s+def|def|func|function)\s", re.M)


@dataclass(frozen=True)
class Query:
    text: str
    terms: frozenset[str]
    anchors: frozenset[str]
    documentation: bool
    languages: frozenset[str]


def query(requirement):
    base = " ".join([requirement["skill"], *requirement["aliases"]])
    obligation = " ".join([requirement["quote"], requirement["depth"]])
    base_terms = set(tokenize(base, 4096))
    terms = list(dict.fromkeys(tokenize(obligation + " " + base, 4096)))
    content = [t for t in terms if t not in _STOP]
    if not content:
        content = list(dict.fromkeys(tokenize(base, 4096)))
    included = []
    for term in content:
        if len(" ".join([*included, term])) > 256:
            break
        included.append(term)
    chosen = frozenset(included)
    anchors = chosen - base_terms
    specific = anchors - _GENERIC_OBJECTS
    return Query(
        " ".join(included),
        chosen,
        frozenset(specific or anchors),
        bool(set(tokenize(obligation, 4096)) & {"documentation", "documented", "readme"}),
        frozenset(
            s
            for label in [requirement["skill"], *requirement["aliases"]]
            for s in _LANGUAGES.get(label.casefold().strip(), set())
        ),
    )


def window(candidate, request):
    """Select an exact contiguous <=8-line/300-byte window, never a partial line.

    Score only text actually retained, not path/chunk matches outside the window.
    A specific object/action must overlap when present; technology-only noise is
    insufficient. Unknown semantic equivalence remains a bounded-search limitation.
    """
    suffix = PurePosixPath(candidate.path.lower()).suffix
    if request.languages and suffix in _CODE_SUFFIXES and suffix not in request.languages:
        return None
    lines = candidate.excerpt.splitlines()
    line_terms = [set(tokenize(line, 16384)) for line in lines]
    best = None
    for start in range(len(lines)):
        kept, terms = [], set()
        for end in range(start, min(start + 8, len(lines))):
            proposed = "\n".join([*kept, lines[end]])
            if len(proposed.encode()) > 300:
                break
            kept.append(lines[end])
            terms.update(line_terms[end])
            hits = terms & request.terms
            anchors = terms & request.anchors
            if not hits or (request.anchors and not anchors):
                continue
            excerpt = "\n".join(kept)
            kind = career_evidence.kind(candidate.path, excerpt)
            preferred = kind == ("documentation" if request.documentation else "code")
            # Prefer an intact small callable over a disembodied tail/header.
            # This is a window-quality heuristic, NOT proof of implementation support.
            declaration = _CALLABLE.search(excerpt)
            prefix = excerpt[: declaration.start()] if declaration else ""
            body = excerpt[declaration.start() :] if declaration else ""
            clean_prefix = all(
                not line.strip() or line.lstrip().startswith(("//", "#"))
                for line in prefix.splitlines()
            )
            complete = (
                bool(declaration)
                and clean_prefix
                and (
                    suffix == ".py"
                    and kind == "code"
                    or "{" in body
                    and body.count("{") == body.count("}")
                    and body.rstrip().endswith("}")
                )
            )
            score = (
                preferred,
                complete,
                len(anchors),
                len(hits),
                sum(bool(line.strip()) for line in kept),
                -start,
            )
            if best is None or score > best[0]:
                best = (score, start, end, excerpt)
    return best
