"""Reject explicit literal conflicts; no execution, temporal matching or control-flow proof."""

import ast
import math
import re
import textwrap

_VALUE = r"(?:True|False|None|-?\d+(?:\.\d+)?|'[^'\n]{0,80}'|\"[^\"\n]{0,80}\")"
_ASSIGNMENT = re.compile(
    r"\b([A-Za-z_]\w*)\s+(?:is|equals|defaults to)\s+("
    + _VALUE
    + r")|\b([A-Za-z_]\w*)\s*=\s*("
    + _VALUE
    + r")"
)
_RETURN = re.compile(r"\b([A-Za-z_]\w*)\s+returns\s+(" + _VALUE + r")")


def key(value):
    if type(value) in (int, float) and (type(value) is int or math.isfinite(value)):
        return ("number", value)
    if type(value) in (bool, str, type(None)):
        return (type(value).__name__, value)
    return None


def parsed_key(text):
    try:
        return key(ast.literal_eval(text))
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Unsupported literal") from exc


def literal_conflict(claim, evidence):
    try:
        tree = ast.parse(textwrap.dedent(evidence))
    except (SyntaxError, ValueError, RecursionError):
        return False
    assignments, returns = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    value = key(node.value.value) if isinstance(node.value, ast.Constant) else None
                    assignments.setdefault(target.id, set()).add(value)
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            # Only an undecorated single literal return: never infer arbitrary ordering,
            # branch effects, or behavior through calls, imports or exception paths.
            value = None
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and isinstance(body[0].value, ast.Constant)
            ):
                value = key(body[0].value.value)
            returns.setdefault(node.name, set()).add(value)
    for match in _ASSIGNMENT.finditer(claim):
        name, value, other_name, other_value = match.groups()
        source = assignments.get(name or other_name, set())
        expected = parsed_key(value or other_value)
        if len(source) == 1 and None not in source and expected not in source:
            return True
    for name, value in _RETURN.findall(claim):
        source = returns.get(name, set())
        if len(source) == 1 and None not in source and parsed_key(value) not in source:
            return True
    return False


def literal_facts(evidence):
    """Tiny definition-level facts only; never execute code or analyze branching/order."""
    try:
        tree = ast.parse(textwrap.dedent(evidence))
    except (SyntaxError, ValueError, RecursionError):
        return []
    facts = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and key(node.value.value) is not None:
                    facts.append(f"Declaration {target.id} = {node.value.value!r}")
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            value = body[0].value
            if isinstance(value, ast.Constant) and key(value.value) is not None:
                facts.append(f"Definition {node.name} body returns {value.value!r}")
            if (
                isinstance(value, ast.UnaryOp)
                and isinstance(value.op, ast.Not)
                and isinstance(value.operand, ast.Name)
                and len(node.args.posonlyargs + node.args.args) == 1
                and value.operand.id == (node.args.posonlyargs + node.args.args)[0].arg
                and not node.args.vararg
                and not node.args.kwarg
                and not node.args.kwonlyargs
            ):
                name = value.operand.id
                facts.append(
                    f"Definition {node.name}: {name}=False returns True; {name}=True returns False"
                )
        if len(facts) >= 4:
            break
    return [fact for fact in facts[:4] if len(fact) <= 100]
