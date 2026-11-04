"""Bounded local implementation groups over an already authorized canonical snapshot.

Static hints only: no execution, fetching, model, index or semantic proof.
"""

import ast
import builtins
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from core.services import career_evidence
from rag.generator.grounding import unsafe
from rag.retriever.revision_keyword import tokenize

MAX_ROUTINES = 256
MAX_DEPTH = 3
MAX_CLOSURE = 8
_BUILTINS = {
    "if",
    "for",
    "switch",
    "func",
    "make",
    "new",
    "len",
    "cap",
    "append",
    "copy",
    "delete",
    "close",
    "int",
    "int64",
    "float64",
    "string",
    "bool",
    "byte",
    "rune",
    "list",
    "tuple",
    "dict",
    "set",
    "str",
    "float",
    "range",
    "enumerate",
    "zip",
    "isinstance",
    "print",
    "super",
    "min",
    "max",
    "sum",
    "sorted",
    "error",
}
_PY_BUILTINS = frozenset(n for n in dir(builtins) if not n.startswith("_"))
_GO_DEF = re.compile(r"(?m)^func\s+(?:\([^\n]*?\)\s+)?(\w+)\s*\(")
# Definition text anywhere in the revision, including files that could not be parsed or grouped.
# A call to a name defined here is local code: it must be resolved and shown, never assumed.
_DEFINED = re.compile(
    r"(?m)^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)"
    r"|^[ \t]*func[ \t]+(?:\([^)\n]*\)[ \t]*)?(\w+)[ \t]*[\[(]"
)
_CALL = re.compile(r"\b(?:(\w+)\s*\.\s*)?(\w+)\s*\(")


@dataclass(frozen=True)
class Routine:
    path: str
    name: str
    start: int
    end: int
    # name, explicit local module path (or None), external namespace, call kind:
    # "attr" (obj.name()), "name" (bare), "import" (absolute from-import), "relative".
    calls: tuple
    search: str
    stub: bool = False
    delegation: bool = False
    bindings: tuple = ()
    values: tuple = ()  # (imported name, module path, import span) read as values


@dataclass(frozen=True)
class Window:
    source_id: object
    chunk_id: object
    path: str
    commit_sha: str
    line_start: int
    line_end: int
    excerpt: str
    implementation_complete: bool | None = None
    is_clipped: bool = False

    def model_dump(self):
        return asdict(self)


@dataclass(frozen=True)
class Group:
    priority: tuple
    windows: tuple
    complete: bool | None
    missing: tuple = ()
    clipped: tuple = ()


def _mask_go(text):
    """Keep offsets/newlines while hiding strings and comments from brace scanning."""
    output, i, state = list(text), 0, None
    while i < len(text):
        c, pair = text[i], text[i : i + 2]
        if state is None:
            if pair in {"//", "/*"}:
                state = pair
                output[i : i + 2] = "  "
                i += 2
                continue
            if c in {'"', "'", "`"}:
                state = c
                output[i] = " "
        elif state == "//" and c == "\n":
            state = None
        elif state == "/*" and pair == "*/":
            output[i : i + 2] = "  "
            state = None
            i += 2
            continue
        elif state in {'"', "'"} and c == "\\":
            output[i] = " "
            if i + 1 < len(text):
                output[i + 1] = " " if text[i + 1] != "\n" else "\n"
            i += 2
            continue
        elif c == state:
            output[i] = " "
            state = None
        if state is not None and c != "\n":
            output[i] = " "
        i += 1
    return "".join(output)


def routines(path, text):
    lines = text.splitlines()
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return []
        imports, external = {}, set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                parent = PurePosixPath(path).parent if node.level else PurePosixPath()
                for _ in range(max(0, node.level - 1)):
                    parent = parent.parent
                module = str(parent / (node.module.replace(".", "/") + ".py"))
                for alias in node.names:
                    imports[alias.asname or alias.name] = (
                        alias.name,
                        module,
                        (node.lineno, node.end_lineno),
                        "relative" if node.level else "import",
                    )
            elif isinstance(node, ast.Import):
                external.update(a.asname or a.name.split(".")[0] for a in node.names)
        found = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls, bindings = [], []
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Name):
                    name = call.func.id
                    target, module, binding, kind = imports.get(name, (name, None, None, "name"))
                    parameters = {
                        a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    }
                    if name in parameters:
                        module, kind = ":dynamic_parameter", "name"
                    if binding:
                        bindings.append(binding)
                    calls.append((target, module, False, kind))
                elif isinstance(call.func, ast.Attribute):
                    prefix = call.func.value.id if isinstance(call.func.value, ast.Name) else "?"
                    calls.append((call.func.attr, None, prefix in external, "attr"))
            values = sorted(
                {
                    (imports[ref.id][0], imports[ref.id][1], imports[ref.id][2])
                    for ref in ast.walk(node)
                    if isinstance(ref, ast.Name)
                    and isinstance(ref.ctx, ast.Load)
                    and ref.id in imports
                }
            )
            body = [
                n
                for n in node.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)
                )
            ]
            stub = not body or any(
                isinstance(n, ast.Pass)
                or isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and n.value.value is Ellipsis
                or isinstance(n, ast.Raise)
                and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name)
                and n.exc.func.id == "NotImplementedError"
                for n in body
            )
            delegation = (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and isinstance(body[0].value, ast.Call)
            )
            found.append(
                Routine(
                    path,
                    node.name,
                    min([node.lineno, *(d.lineno for d in node.decorator_list)]),
                    node.end_lineno,
                    tuple(calls),
                    "\n".join(
                        lines[
                            min([node.lineno, *(d.lineno for d in node.decorator_list)])
                            - 1 : node.end_lineno
                        ]
                    ),
                    stub,
                    delegation,
                    tuple(sorted(set(bindings))),
                    tuple(values),
                )
            )
        return found
    if not path.endswith(".go"):
        return []
    masked = _mask_go(text)
    imports = set()
    for block in re.finditer(r"(?ms)^\s*import\s*(?:\((.*?)\)|([^\n]+))", text):
        for match in re.finditer(r'(?:(\w+)\s+)?"([^"\n]+)"', block.group(1) or block.group(2)):
            imports.add(match.group(1) or match.group(2).rsplit("/", 1)[-1])
    found = []
    for match in _GO_DEF.finditer(masked):
        parens, start, i = 0, None, match.end() - 1
        while i < len(masked):
            c = masked[i]
            parens += (c == "(") - (c == ")")
            if c == "{" and parens == 0:
                before = masked[match.end() : i].rstrip()
                if before.endswith("interface") or before.endswith("struct"):
                    j, depth = i + 1, 1
                    while j < len(masked) and depth:
                        depth += (masked[j] == "{") - (masked[j] == "}")
                        j += 1
                    i = j
                    continue
                start = i
                break
            i += 1
        if start is None:
            continue
        depth, end = 1, start + 1
        while end < len(masked) and depth:
            depth += (masked[end] == "{") - (masked[end] == "}")
            end += 1
        if depth:
            continue
        first, last = text[: match.start()].count("\n") + 1, text[:end].count("\n") + 1
        leading = first - 2
        while leading >= 0 and lines[leading].lstrip().startswith("//"):
            leading -= 1
        search = "\n".join(lines[leading + 1 : last])
        body = masked[start + 1 : end - 1]
        dynamic = set(re.findall(r"\b(\w+)\s+func\s*\(", masked[match.end() : start]))
        calls = tuple(
            (
                name,
                ":dynamic_parameter" if name in dynamic else None,
                prefix in imports,
                "attr" if prefix else "name",
            )
            for prefix, name in _CALL.findall(body)
            if name not in {"if", "for", "switch", "func"}
        )
        delegation = bool(re.fullmatch(r"\s*return\s+[\w.]+\s*\([^;]*\)\s*;?\s*", body))
        found.append(
            Routine(path, match.group(1), first, last, calls, search, not body.strip(), delegation)
        )
    return found


def _value_span(files, module, name):
    """Line span of a top-level `name = value` in a parsed, safe local code module, else None."""
    file = files.get(module)
    if (
        not module
        or not module.endswith(".py")
        or file is None
        or "text" not in file
        or unsafe(file["text"])
        or any(p.startswith(".env") for p in module.lower().split("/"))
        or career_evidence.kind(module, file["text"]) != "code"
    ):
        return None
    try:
        tree = ast.parse(file["text"])
    except (SyntaxError, ValueError, RecursionError):
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            or (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            )
        ):
            return node.lineno, node.end_lineno
    return None


def groups(snapshot, seeds, request, hints=(), retrieved=frozenset()):
    """Resolve finite local calls inside THIS verified revision; never another tenant/repo.

    `retrieved` holds chunk IDs returned by semantic/hybrid retrieval. A routine containing one
    is eligible without lexical term overlap (bridging vocabulary is the purpose of semantic
    recall); its completeness, closure and size rules are unchanged.
    """
    files = {}
    for chunk, path in snapshot.chunks:
        file = files.setdefault(path, {"lines": {}, "chunks": {}})
        for number, line in enumerate(chunk.text.splitlines(), chunk.line_start):
            if number in file["lines"] and file["lines"][number] != line:
                raise ValueError("Inconsistent canonical overlap")
            file["lines"][number] = line
            file["chunks"].setdefault(number, chunk)
    defined = {
        name
        for chunk, _ in snapshot.chunks
        for match in _DEFINED.finditer(chunk.text)
        for name in match.groups()
        if name
    }
    paths = {path for _, path in snapshot.chunks}
    inventory = []
    for path, file in sorted(files.items()):
        numbers = sorted(file["lines"])
        if numbers != list(range(1, len(numbers) + 1)):
            continue
        file["text"] = "\n".join(file["lines"][n] for n in numbers)
        if unsafe(file["text"]) or any(p.startswith(".env") for p in path.lower().split("/")):
            continue
        inventory.extend(routines(path, file["text"]))
        if len(inventory) > MAX_ROUTINES:
            return [], {
                "routine_limit": True,
                "budget_conflicts": 0,
                "parsed_chunks": {
                    str(c.id) for c, p in snapshot.chunks if p.endswith((".go", ".py"))
                },
            }
    parsed_chunks = {
        str(files[u.path]["chunks"][n].id) for u in inventory for n in range(u.start, u.end + 1)
    }
    seed_ids = {str(c.chunk_id) for c in seeds}
    output, conflicts = [], 0
    for root in inventory:
        suffix = PurePosixPath(root.path).suffix
        if request.languages and suffix not in request.languages:
            continue
        file = files[root.path]
        if not any(str(file["chunks"][n].id) in seed_ids for n in range(root.start, root.end + 1)):
            continue
        terms = set(tokenize(root.search, 16384))
        hint = next(
            (
                h
                for h in hints
                if h.path == root.path and h.start == root.start and h.end == root.end
            ),
            None,
        )
        semantic_seed = any(
            str(file["chunks"][n].id) in retrieved for n in range(root.start, root.end + 1)
        )
        if (
            hint is None
            and not semantic_seed
            and (not terms & request.terms or request.anchors and not terms & request.anchors)
        ):
            continue
        closure, missing = [], []

        def visit(unit, depth, stack):
            key = (unit.path, unit.start)
            if key in stack:
                missing.append("recursive_call")
                return
            if key in {(u.path, u.start) for u in closure}:
                return
            if depth > MAX_DEPTH or len(closure) >= MAX_CLOSURE:
                missing.append("call_followup_limit")
                return
            closure.append(unit)
            if unit.stub:
                missing.append("stub_implementation")
            for name, module, external, kind in unit.calls:
                if external:
                    if unit.delegation:
                        missing.append("opaque_external_call:" + name)
                    continue
                candidates = [
                    u
                    for u in inventory
                    if u.name == name
                    and not u.path.endswith("_test.go")
                    and (
                        u.path == unit.path
                        or career_evidence.kind(u.path, files[u.path]["text"]) == "code"
                    )
                    and (
                        u.path == module
                        if module
                        else PurePosixPath(u.path).parent == PurePosixPath(unit.path).parent
                        and PurePosixPath(u.path).suffix == PurePosixPath(unit.path).suffix
                    )
                ]
                same = [u for u in candidates if u.path == unit.path]
                candidates = same or candidates
                python = unit.path.endswith(".py")
                if (
                    not candidates
                    and module is None
                    and (name in _BUILTINS or python and kind == "name" and name in _PY_BUILTINS)
                ):
                    continue  # A verified local shadow is resolved before assuming a builtin.
                if (
                    not candidates
                    and name not in defined
                    and (kind == "attr" or kind == "import" and module not in paths)
                ):
                    # Defined nowhere in this revision: non-local code (builtin methods, stdlib or
                    # third-party objects, framework decorators). Same rule as an imported
                    # external namespace; a pure delegation to it is never implementation proof.
                    if unit.delegation:
                        missing.append("opaque_external_call:" + name)
                    continue
                if len(candidates) != 1:
                    missing.append("unresolved_or_ambiguous_call:" + name)
                else:
                    visit(candidates[0], depth + 1, stack | {key})

        visit(root, 0, set())
        # Values (e.g. thresholds) imported from a local module: show the import and the defining
        # top-level assignment. Purely additive; an unresolvable value is left exactly as before.
        extra = {}
        for unit in list(closure):
            for name, module, binding in unit.values:
                span = _value_span(files, module, name)
                if span is None or len(closure) >= MAX_CLOSURE:
                    continue
                extra.setdefault((unit.path, unit.start), set()).add(binding)
                if not any(u.path == module and u.start == span[0] for u in closure):
                    closure.append(Routine(module, name, span[0], span[1], (), ""))
        windows = []
        for unit in closure:
            file = files[unit.path]
            support = hint.bindings if hint is not None and unit == root else ()
            spans = sorted(
                [
                    *unit.bindings,
                    *support,
                    *extra.get((unit.path, unit.start), ()),
                    (unit.start, unit.end),
                ]
            )
            merged = []
            for left, right in spans:
                if merged and left <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            for left, right in merged:
                n = left
                while n <= right:
                    chunk, kept, start = file["chunks"][n], [], n
                    while n <= right and len(kept) < 8 and file["chunks"][n].id == chunk.id:
                        line = file["lines"][n]
                        if len("\n".join([*kept, line]).encode()) > 300:
                            break
                        kept.append(line)
                        n += 1
                    if not kept:
                        missing.append("oversized_source_line")
                        break
                    windows.append(
                        Window(
                            chunk.source_id,
                            chunk.id,
                            unit.path,
                            snapshot.sha,
                            start,
                            n - 1,
                            "\n".join(kept),
                            is_clipped=(start != chunk.line_start or n - 1 != chunk.line_end),
                        )
                    )
            if missing and missing[-1] == "oversized_source_line":
                break
        if len(windows) > 5 or any(m == "oversized_source_line" for m in missing):
            conflicts += 1
            continue  # Never silently truncate a claimed complete implementation.
        complete = not missing
        windows = tuple(
            Window(**{**asdict(w), "implementation_complete": complete}) for w in windows
        )
        priority = (
            not request.documentation,
            complete,
            len((terms | (hint.features if hint else frozenset())) & request.anchors),
            len((terms | (hint.features if hint else frozenset())) & request.terms),
        )
        output.append(Group(priority, windows, complete, tuple(sorted(set(missing)))))
    output.sort(
        key=lambda group: (
            group.priority,
            -len(group.windows),
            tuple((w.path, -w.line_start) for w in group.windows),
        ),
        reverse=True,
    )
    return output, {
        "routine_limit": False,
        "budget_conflicts": conflicts,
        "parsed_chunks": parsed_chunks,
    }
