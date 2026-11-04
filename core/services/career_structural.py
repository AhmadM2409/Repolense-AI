"""Bounded deterministic structural retrieval hints, never capability verdicts.

A narrow source-derived index bridges HTTP/guard vocabulary to actual constructs.
No vectors, models, generated evidence, query rewriting, source execution or persistence.
"""

import ast
import re
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from core.services.career_evidence import unsafe

MAX_FILES = 32
MAX_FILE_BYTES = 32768
MAX_TOTAL_BYTES = 262144
MAX_HINTS = 128
MAX_SEEDS = 2
_HTTP = frozenset("http api endpoint handler request web route controller inbound".split())
_GUARD = frozenset(
    "validation validate validating check checking reject rejecting guard verify screen".split()
)
_METHODS = {"get", "post", "put", "patch", "delete", "route"}


@dataclass(frozen=True)
class Hint:
    path: str
    start: int
    end: int
    symbol: str
    bindings: tuple
    features: frozenset


@dataclass(frozen=True)
class Candidate:
    source_id: object
    chunk_id: object
    path: str
    commit_sha: str
    line_start: int
    line_end: int
    excerpt: str

    def model_copy(self, *, update):
        return replace(self, **update)


def eligible(request):
    return bool(request.terms & _HTTP and request.terms & _GUARD and not request.documentation)


def excluded_asset(path, request):
    """Styles/vector assets are not server-side request guards.

    Keep ordinary lexical behavior for other job requirements and preserve actual
    documentation/configuration candidates. This filter never assigns support.
    """
    return bool(
        request is not None
        and eligible(request)
        and PurePosixPath(path.lower()).suffix in {".css", ".scss", ".sass", ".less", ".svg"}
    )


def _python(path, text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    constructors, request_names, abort_names = {}, set(), set()
    imports = {}
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = (node.lineno, node.end_lineno)
                if (node.module, alias.name) in {
                    ("fastapi", "FastAPI"),
                    ("flask", "Flask"),
                    ("flask", "Blueprint"),
                }:
                    constructors[name] = (node.lineno, node.end_lineno)
                if (node.module, alias.name) == ("flask", "request"):
                    request_names.add(name)
                if (node.module, alias.name) == ("flask", "abort"):
                    abort_names.add(name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node)
    receivers = {}
    for name, values in assignments.items():
        if len(values) != 1:
            continue
        node = values[0]
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in constructors
            and value.func.id not in assignments
        ):
            receivers[name] = (constructors[value.func.id], (node.lineno, node.end_lineno))
    hints = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bindings = []
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in _METHODS
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id in receivers
            ):
                bindings.extend(receivers[decorator.func.value.id])
        if not bindings:
            continue
        inputs = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                value = child.value.func if isinstance(child.value, ast.Call) else child.value
                if (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id in request_names
                    and value.attr in {"get_json", "json", "form", "args", "values"}
                ):
                    inputs.update(t.id for t in child.targets if isinstance(t, ast.Name))
                    bindings.append(imports[value.value.id])
        guarded = False
        for child in ast.walk(node):
            if not isinstance(child, ast.If) or not inputs & {
                n.id for n in ast.walk(child.test) if isinstance(n, ast.Name)
            }:
                continue
            for branch in child.body:
                for action in ast.walk(branch):
                    if isinstance(action, ast.Raise):
                        guarded = True
                    if (
                        isinstance(action, ast.Call)
                        and isinstance(action.func, ast.Name)
                        and action.func.id in abort_names
                        and action.func.id not in inputs
                        and any(
                            isinstance(a, ast.Constant)
                            and isinstance(a.value, int)
                            and 400 <= a.value < 500
                            for a in action.args
                        )
                    ):
                        bindings.append(imports[action.func.id])
                        guarded = True
        if guarded:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            hints.append(
                Hint(
                    path,
                    start,
                    node.end_lineno,
                    node.name,
                    tuple(sorted(set(bindings))),
                    _HTTP | _GUARD,
                )
            )
    return hints


def _go(path, text):
    from core.services.career_context import _mask_go, routines

    imported = re.search(r'(?m)^\s*(?:import\s+)?(?:(\w+)\s+)?"net/http"', text)
    if not imported:
        return []
    alias = imported.group(1) or "http"
    if alias in {"_", "."}:
        return []
    # Accept a single import or a closed import block, never invent an import binding.
    import_line = text[: imported.start() + imported.group(0).index('"net/http"')].count("\n") + 1
    block = re.search(r"(?ms)^import\s*\(.*?\)", text)
    if block and block.start() <= imported.start() < block.end():
        binding = (text[: block.start()].count("\n") + 1, text[: block.end()].count("\n") + 1)
    elif re.search(r'(?m)^import\s+(?:\w+\s+)?"net/http"', text):
        binding = (import_line, import_line)
    else:
        return []
    masked = _mask_go(text)
    lines = text.splitlines()
    hints = []
    for unit in routines(path, text):
        segment = "\n".join(lines[unit.start - 1 : unit.end])
        signature = segment.split("{", 1)[0]
        arg = re.search(r"\b(\w+)\s+\*\s*" + re.escape(alias) + r"\.Request\b", signature)
        if not arg or not re.search(re.escape(alias) + r"\.ResponseWriter\b", signature):
            continue
        body = "\n".join(masked.splitlines()[unit.start - 1 : unit.end])
        # Bounded, conservative structural candidate: request-derived guard plus
        # explicit client-error response and return. This is not a data-flow proof.
        guard = re.search(
            r"\bif\b[^{}]*\b" + re.escape(arg.group(1)) + r"\.[^{}]*\{([^{}]*)\}", body
        )
        if (
            guard
            and re.search(re.escape(alias) + r"\.Error\s*\([^;]*\b4\d\d\s*\)", guard.group(1))
            and re.search(r"\breturn\b", guard.group(1))
        ):
            hints.append(Hint(path, unit.start, unit.end, unit.name, (binding,), _HTTP | _GUARD))
    return hints


def discover(snapshot, request):
    stats = {
        "structural_scanned_files": 0,
        "structural_scanned_bytes": 0,
        "structural_hints": 0,
        "structural_scan_limited": False,
    }
    if not eligible(request):
        return [], [], stats
    files = {}
    for chunk, path in snapshot.chunks:
        if PurePosixPath(path).suffix not in {".py", ".go"}:
            continue
        file = files.setdefault(path, {"lines": {}, "chunks": {}})
        for number, line in enumerate(chunk.text.splitlines(), chunk.line_start):
            if number in file["lines"] and file["lines"][number] != line:
                raise ValueError("Inconsistent structural canonical overlap")
            file["lines"][number] = line
            file["chunks"].setdefault(number, chunk)
    hints = []
    for path, file in sorted(files.items()):
        numbers = sorted(file["lines"])
        if numbers != list(range(1, len(numbers) + 1)):
            continue
        text = "\n".join(file["lines"][n] for n in numbers)
        size = len(text.encode())
        if (
            stats["structural_scanned_files"] >= MAX_FILES
            or size > MAX_FILE_BYTES
            or stats["structural_scanned_bytes"] + size > MAX_TOTAL_BYTES
        ):
            stats["structural_scan_limited"] = True
            continue
        stats["structural_scanned_files"] += 1
        stats["structural_scanned_bytes"] += size
        if unsafe(text) or any(p.startswith(".env") for p in path.split("/")):
            continue
        found = _python(path, text) if path.endswith(".py") else _go(path, text)
        for hint in found:
            if request.languages and PurePosixPath(path).suffix not in request.languages:
                continue
            if len(hints) >= MAX_HINTS:
                stats["structural_scan_limited"] = True
                break
            hints.append(hint)
    stats["structural_hints"] = len(hints)
    seeds = []
    chosen = []
    for hint in hints:
        chunk = files[hint.path]["chunks"].get(hint.start)
        if chunk is None:
            continue
        if str(chunk.id) not in {str(c.chunk_id) for c in seeds}:
            if len(seeds) >= MAX_SEEDS:
                stats["structural_scan_limited"] = True
                continue
            seeds.append(
                Candidate(
                    chunk.source_id,
                    chunk.id,
                    hint.path,
                    snapshot.sha,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.text,
                )
            )
        chosen.append(hint)
    return seeds, chosen, stats
