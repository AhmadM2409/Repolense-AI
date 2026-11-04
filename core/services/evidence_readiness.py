"""Conservative evidence availability for finite static source patterns; never execution."""

import ast
import re
from dataclasses import dataclass

from rag.generator.completeness import requirements


@dataclass(frozen=True)
class Readiness:
    ready: bool
    missing: tuple[str, ...] = ()
    reason: str = "unsupported_or_missing_requirements"


def assess(objective, matches):
    if not matches:
        return Readiness(False, reason="no_evidence")
    definitions = {}
    imports = {}
    ambiguous = set()
    for item in matches:
        if not item.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(item.excerpt)
        except (SyntaxError, ValueError, RecursionError):
            continue
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [n.id for n in node.targets if isinstance(n, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    imports[(item.path, alias.asname or alias.name)] = (
                        node.module.replace(".", "/") + ".py",
                        alias.name,
                    )
            for name in names:
                key = (item.path, name)
                if key in definitions:
                    ambiguous.add(key)
                definitions[key] = node
    mentioned = [
        key for key in definitions if re.search(r"\b" + re.escape(key[1]) + r"\b", objective)
    ]
    if not mentioned:
        return Readiness(False)
    if len({name for _, name in mentioned}) != len(mentioned):
        return Readiness(False, reason="ambiguous_target")
    for unit in requirements(objective):
        text = unit["text"]
        if not re.search(
            r"\b(describe|return|returns|literal|numeric|threshold|condition|compare|"
            r"comparison|value|values|larger|smaller|greater|less)\b",
            text,
            re.I,
        ):
            return Readiness(False)

        names = {key[1] for key in definitions} | {key[1] for key in imports}
        anchored = any(re.search(r"\b" + re.escape(name) + r"\b", text) for name in names)
        relative = re.search(r"\b(its|their|which)\b", text, re.I)
        if not anchored and not relative:
            return Readiness(False)
    missing = set()
    numeric = False

    def constant(node):
        if isinstance(node, ast.Constant):
            return node
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in {int, float}
        ):
            value = node.operand.value
            return ast.Constant(value=-value if isinstance(node.op, ast.USub) else value)
        return None

    def literal(path, name):
        key = imports.get((path, name), (path, name))
        if key in ambiguous:
            return None
        node = definitions.get(key)
        if node is None:
            if (path, name) in imports:
                missing.add(key[1])
            return None
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return constant(node.value)
        return None

    for key in mentioned:
        if key in ambiguous:
            return Readiness(False)
        node = definitions[key]
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = constant(node.value)
            if value is None:
                return Readiness(False)
            numeric |= type(value.value) in {int, float}
            continue
        if (
            not isinstance(node, ast.FunctionDef)
            or node.decorator_list
            or len(node.body) != 1
            or not isinstance(node.body[0], ast.Return)
        ):
            return Readiness(False)
        expression = node.body[0].value
        if not isinstance(expression, (ast.Constant, ast.Compare, ast.Name, ast.UnaryOp)):
            return Readiness(False)
        parameters = {a.arg for a in node.args.posonlyargs + node.args.args}
        if re.search(r"\b(literal|string)\b", objective, re.I):
            value = constant(expression)
            if (
                value is None
                and isinstance(expression, ast.Name)
                and expression.id not in parameters
            ):
                value = literal(key[0], expression.id)
            if value is None and not missing:
                return Readiness(False)
        if re.search(r"\b(compare|comparison|condition)\b", objective, re.I) and not isinstance(
            expression, ast.Compare
        ):
            return Readiness(False)

        for part in ast.walk(expression):
            if isinstance(part, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp)):
                return Readiness(False)
            if isinstance(part, ast.Constant):
                numeric |= type(part.value) in {int, float}
            if isinstance(part, ast.Name) and part.id not in parameters:
                value = literal(key[0], part.id)
                if value is None and not missing:
                    return Readiness(False)
                if value is not None:
                    numeric |= type(value.value) in {int, float}
    if missing:
        return Readiness(False, tuple(sorted(missing)), "missing_imported_definition")
    if (
        re.search(r"\b(numeric|threshold|number|how many|how much)\b", objective, re.I)
        and not numeric
    ):
        return Readiness(False)
    return Readiness(True, reason="bounded_source_available_final_validation_required")
