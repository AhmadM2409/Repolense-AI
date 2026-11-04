"""Versioned Python AST rules. Matches are observations or potential risks, not exploits."""

import ast
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

VERSION = "deterministic-v1"
RULES = {
    "PY-BARE-EXCEPT": (
        "Bare exception handler",
        "medium",
        "maintainability",
        "observation",
        "A bare except handler catches BaseException, including interruption signals.",
    ),
    "PY-SHELL-TRUE": (
        "Subprocess shell enabled",
        "high",
        "security",
        "potential",
        (
            "A subprocess call explicitly enables shell=True. Review whether command "
            "data is trusted; exploitability is not established."
        ),
    ),
    "PY-SQL-DYNAMIC": (
        "Dynamic SQL construction",
        "medium",
        "security",
        "potential",
        (
            "SQL text is assembled dynamically at an execute call. Review "
            "parameterization and input trust; no injection is proven."
        ),
    ),
    "PY-DESERIALIZE": (
        "Potentially unsafe deserialization",
        "high",
        "security",
        "potential",
        (
            "Deserialization can construct objects or interpret untrusted data. Review "
            "input provenance and loader/version behavior."
        ),
    ),
    "PY-DEV-CONFIG": (
        "Development configuration enabled",
        "medium",
        "configuration",
        "potential",
        (
            "An explicit development/debug option is enabled outside a recognized main "
            "guard. Deployment context is unknown."
        ),
    ),
    "PY-SECRET-LITERAL": (
        "Potential credential literal",
        "high",
        "security",
        "potential",
        (
            "A credential-named assignment contains a non-placeholder high-entropy or "
            "recognized-prefix literal. The value has not been validated as a "
            "credential."
        ),
    ),
}


@dataclass(frozen=True)
class Hit:
    rule_id: str
    start: int
    end: int


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = name(node.value)
        return parent + "." + node.attr if parent else ""
    return ""


def main_guard(node):
    test = node.test if isinstance(node, ast.If) else None
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def credential(value):
    if not isinstance(value, str) or len(value) < 20 or len(value) > 512:
        return False
    if re.search(
        r"example|placeholder|change.?me|dummy|fixture|your[_ -]|not.?a.?|test|[<>{}]", value, re.I
    ):
        return False
    entropy = -sum(
        (value.count(c) / len(value)) * math.log2(value.count(c) / len(value)) for c in set(value)
    )
    recognized = re.match(r"^(gh[pousr]_|github_pat_|sk-(?:proj-)?|AKIA[0-9A-Z]{16})", value)
    classes = sum(
        bool(re.search(pattern, value))
        for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^a-zA-Z0-9]")
    )
    return (bool(recognized) and entropy >= 3.0) or (entropy >= 3.5 and classes >= 3)


def python_rules(path, text):
    tree = ast.parse(text, filename=path)  # Parsing only; never compile/exec/import.
    nodes = list(ast.walk(tree))
    if len(nodes) > 50000:
        raise ValueError("Python AST node limit exceeded")
    bindings = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.asname or alias.name.split(".")[0]
                bindings.setdefault(key, set()).add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    target = node.module + "." + alias.name if node.level == 0 else "(relative)"
                    bindings.setdefault(alias.asname or alias.name, set()).add(target)
    imports = {
        key: next(iter(values))
        for key, values in bindings.items()
        if len(values) == 1 and "(relative)" not in values
    }
    # Conservative suppression: an import alias rebound anywhere makes resolution ambiguous.
    rebound = {
        node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    rebound |= {
        name(node).split(".")[0]
        for node in nodes
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
    }
    rebound |= {node.arg for node in nodes if isinstance(node, ast.arg)}
    rebound |= {
        node.name
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    imports = {key: value for key, value in imports.items() if key not in rebound}

    def resolve(node):
        raw = name(node)
        first, _, suffix = raw.partition(".")
        return imports.get(first, "") + ("." + suffix if suffix and first in imports else "")

    guarded = set()
    for node in nodes:
        if main_guard(node):
            guarded.update(id(child) for part in node.body for child in ast.walk(part))
    flask_apps = set()
    for node in nodes:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and resolve(node.value.func) == "flask.Flask"
        ):
            flask_apps.update(t.id for t in node.targets if isinstance(t, ast.Name))
    test_path = any(
        part.lower() in {"tests", "test", "fixtures", "__tests__"}
        for part in PurePosixPath(path).parts
    ) or PurePosixPath(path).name.startswith("test_")
    hits = []

    def add(rule, node):
        hits.append(
            Hit(
                rule,
                node.lineno,
                node.lineno if rule == "PY-BARE-EXCEPT" else (node.end_lineno or node.lineno),
            )
        )

    for node in nodes:
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            add("PY-BARE-EXCEPT", node)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if (
                not test_path
                and any(
                    re.search(r"(^|_)(api_?key|secret|token|password)(_|$)", name(target), re.I)
                    for target in targets
                )
                and isinstance(node.value, ast.Constant)
                and credential(node.value.value)
            ):
                add("PY-SECRET-LITERAL", node)
            if (
                not test_path
                and PurePosixPath(path).name == "settings.py"
                and any(name(t) == "DEBUG" for t in targets)
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
                and id(node) not in guarded
            ):
                add("PY-DEV-CONFIG", node)
        if not isinstance(node, ast.Call):
            continue
        target = resolve(node.func)
        keywords = {item.arg: item.value for item in node.keywords if item.arg}

        def true(key, values=keywords):
            return isinstance(values.get(key), ast.Constant) and values[key].value is True

        if target in {
            "subprocess.run",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
        } and true("shell"):
            add("PY-SHELL-TRUE", node)
        if target in {"pickle.load", "pickle.loads", "marshal.load", "marshal.loads"}:
            add("PY-DESERIALIZE", node)
        if target == "yaml.load" and resolve(keywords.get("Loader")) not in {
            "yaml.SafeLoader",
            "yaml.CSafeLoader",
        }:
            add("PY-DESERIALIZE", node)
        if (
            not test_path
            and id(node) not in guarded
            and (
                (target == "uvicorn.run" and true("reload"))
                or (
                    isinstance(node.func, ast.Attribute)
                    and name(node.func.value) in flask_apps
                    and node.func.attr == "run"
                    and true("debug")
                )
            )
        ):
            add("PY-DEV-CONFIG", node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"execute", "executemany", "executescript"}
            and name(node.func.value).split(".")[-1]
            in {"cursor", "cur", "conn", "connection", "db", "session"}
            and node.args
        ):
            query = node.args[0]
            if (
                isinstance(query, ast.Call)
                and name(query.func).split(".")[-1] == "text"
                and query.args
            ):
                query = query.args[0]
            if isinstance(query, (ast.JoinedStr, ast.BinOp)):
                literals = " ".join(
                    n.value
                    for n in ast.walk(query)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                )
                dynamic = any(
                    isinstance(n, (ast.FormattedValue, ast.Name, ast.Call, ast.Attribute))
                    for n in ast.walk(query)
                )
                if dynamic and re.search(
                    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b", literals, re.I
                ):
                    add("PY-SQL-DYNAMIC", node)
    return tree, sorted(set(hits), key=lambda hit: (hit.start, hit.rule_id, hit.end))
