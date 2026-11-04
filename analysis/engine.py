"""Pure deterministic analysis of validated persisted documents; no filesystem/network execution."""

import ast
import hashlib
import json
import re
import time
import tomllib
from collections import Counter
from pathlib import PurePosixPath
from uuid import NAMESPACE_URL, uuid5

from analysis.rules import RULES, VERSION, main_guard, python_rules
from core.contracts import EvidenceSpan, FindingContract

LANGUAGES = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".cs": "C#",
    ".c": "C",
    ".cpp": "C++",
    ".rb": "Ruby",
    ".php": "PHP",
    ".kt": "Kotlin",
    ".swift": "Swift",
}
FRAMEWORKS = {
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "react": "React",
    "next": "Next.js",
    "vue": "Vue",
    "express": "Express",
    "pytest": "pytest",
    "jest": "Jest",
    "vitest": "Vitest",
}


class AnalysisLimitError(ValueError):
    pass


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def evidence(document, repository_id, sha, start, end):
    lines = document.text.splitlines()
    if start < 1 or end < start or end > len(lines):
        raise ValueError("Invalid analysis evidence range")
    return EvidenceSpan(
        source_id=document.id,
        repository_id=repository_id,
        commit_sha=sha,
        path=document.path,
        line_start=start,
        line_end=end,
        quote="\n".join(lines[start - 1 : end]),
    )


def analyze(documents, repository_id, sha, review_id, profile_id, *, checkpoint=None):
    deadline = time.monotonic() + 20
    if len(documents) > 100 or sum(len(d.text.encode()) for d in documents) > 4 * 1024 * 1024:
        raise AnalysisLimitError("Source budget exceeded")
    files = []
    deps = []
    frameworks = []
    entrypoints = []
    tests = []
    ci = []
    deploy = []
    docs = []
    diagnostics = []
    findings = []
    languages = Counter()
    directories = Counter()
    test_frameworks = []

    def cite(doc, start=1, end=None):
        if not doc.text.splitlines():
            return []
        return [evidence(doc, repository_id, sha, start, end or start).model_dump(mode="json")]

    def locate(doc, needle):
        line = next(
            (
                i
                for i, value in enumerate(doc.text.splitlines(), 1)
                if needle in value and not value.lstrip().startswith("#")
            ),
            None,
        )
        return cite(doc, line) if line else cite(doc, 1, len(doc.text.splitlines()))

    def pair(doc, key, value, syntax):
        keys = [
            re.escape(key),
            re.escape(json.dumps(key, ensure_ascii=False)),
            re.escape("'" + key + "'"),
        ]
        values = [re.escape(json.dumps(value, ensure_ascii=False)), re.escape("'" + value + "'")]
        pattern = (
            (r"(?m)^[\t ]*(?:" + "|".join(keys) + r")\s*=\s*(?:" + "|".join(values) + r")")
            if syntax == "toml"
            else (
                re.escape(json.dumps(key, ensure_ascii=False))
                + r"\s*:\s*"
                + re.escape(json.dumps(value, ensure_ascii=False))
            )
        )
        match = re.search(pattern, doc.text)
        return (
            cite(
                doc,
                doc.text.count("\n", 0, match.start()) + 1,
                doc.text.count("\n", 0, match.end()) + 1,
            )
            if match
            else cite(doc, 1, len(doc.text.splitlines()))
        )

    def dependency(doc, ecosystem, package, declaration, section, proof=None):
        if len(deps) >= 200:
            raise AnalysisLimitError("Dependency declaration budget exceeded")
        if (
            not isinstance(package, str)
            or not isinstance(declaration, str)
            or max(len(package), len(declaration)) > 255
        ):
            diagnostics.append({"path": doc.path, "code": "unsupported_dependency_declaration"})
            return
        item = {
            "ecosystem": ecosystem,
            "name": package,
            "declaration": declaration,
            "section": section,
            "certainty": "observation",
            "evidence": proof or locate(doc, package),
        }
        deps.append(item)
        normalized = package.lower().replace("_", "-")
        if normalized in FRAMEWORKS:
            frameworks.append(
                {
                    "name": FRAMEWORKS[normalized],
                    "certainty": "inferred",
                    "basis": "dependency declaration; installation/use is not verified",
                    "evidence": item["evidence"],
                }
            )

    for document in sorted(documents, key=lambda d: d.path):
        if checkpoint:
            checkpoint()
        if time.monotonic() > deadline:
            raise AnalysisLimitError("Analysis time budget exceeded")
        path = PurePosixPath(document.path)
        lines = document.text.splitlines()
        language = LANGUAGES.get(path.suffix.lower())
        if language:
            languages[language] += 1
        directory = str(path.parent)
        directories[directory.split("/")[0] if directory != "." else "(root)"] += 1
        files.append(
            {
                "source_id": str(document.id),
                "path": document.path,
                "language": language,
                "lines": len(lines),
                "content_hash": document.content_hash,
            }
        )
        if path.suffix.lower() in {".md", ".rst"} or path.name.lower() in {
            "readme",
            "license",
            "contributing",
            "changelog",
        }:
            docs.append(document.path)
        if any(part.lower() in {"test", "tests", "__tests__"} for part in path.parts) or re.search(
            r"(^test_|\.test\.|_test\.)", path.name
        ):
            tests.append(document.path)
        if document.path.startswith(".github/workflows/") or document.path in {
            ".gitlab-ci.yml",
            ".circleci/config.yml",
        }:
            ci.append(
                {"path": document.path, "certainty": "observation", "evidence": cite(document)}
            )
        if path.name.lower() in {
            "dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml",
            "procfile",
            "vercel.json",
            "render.yaml",
        }:
            deploy.append(
                {"path": document.path, "certainty": "observation", "evidence": cite(document)}
            )
        try:
            if path.name == "package.json":

                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("Duplicate manifest key")
                        result[key] = value
                    return result

                data = json.loads(document.text, object_pairs_hook=unique)
                if not isinstance(data, dict):
                    raise ValueError("Invalid manifest")
                for section in (
                    "dependencies",
                    "devDependencies",
                    "peerDependencies",
                    "optionalDependencies",
                ):
                    section_data = data.get(section, {})
                    if not isinstance(section_data, dict):
                        raise ValueError("Invalid dependency map")
                    for package, value in sorted(section_data.items()):
                        pattern = re.compile(
                            re.escape(json.dumps(package, ensure_ascii=False))
                            + r"\s*:\s*"
                            + re.escape(json.dumps(value, ensure_ascii=False))
                        )
                        match = pattern.search(document.text)
                        proof = (
                            cite(
                                document,
                                document.text.count("\n", 0, match.start()) + 1,
                                document.text.count("\n", 0, match.end()) + 1,
                            )
                            if match
                            else locate(document, package)
                        )
                        dependency(document, "npm", package, value, section, proof)
                if isinstance(data.get("scripts"), dict):
                    for key in ("start", "dev", "serve"):
                        if isinstance(data["scripts"].get(key), str):
                            entrypoints.append(
                                {
                                    "name": key,
                                    "target": data["scripts"][key][:255],
                                    "certainty": "inferred",
                                    "basis": "declared package script, not executed",
                                    "evidence": pair(document, key, data["scripts"][key], "json"),
                                }
                            )
            elif path.name == "pyproject.toml":
                data = tomllib.loads(document.text)
                project = data.get("project", {})
                if not isinstance(project, dict) or not isinstance(
                    project.get("dependencies", []), list
                ):
                    raise ValueError("Invalid project dependency list")
                declarations = [
                    ("project.dependencies", item) for item in project.get("dependencies", [])
                ]
                for group, items in project.get("optional-dependencies", {}).items():
                    if isinstance(items, list):
                        declarations.extend(
                            ("project.optional-dependencies." + group, item) for item in items
                        )
                for section, declaration in declarations:
                    if not isinstance(declaration, str):
                        continue
                    match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", declaration.strip())
                    if match:
                        dependency(
                            document,
                            "python",
                            match[1],
                            match[2] or "(unconstrained)",
                            section,
                            locate(document, declaration),
                        )
                for package, value in (
                    data.get("tool", {}).get("poetry", {}).get("dependencies", {}).items()
                ):
                    if package != "python":
                        dependency(
                            document,
                            "python",
                            package,
                            value if isinstance(value, str) else json.dumps(value, sort_keys=True),
                            "poetry",
                            pair(document, package, value, "toml")
                            if isinstance(value, str)
                            else cite(document, 1, len(lines)),
                        )
                for key, value in project.get("scripts", {}).items():
                    if isinstance(value, str):
                        entrypoints.append(
                            {
                                "name": key[:255],
                                "target": value[:255],
                                "certainty": "inferred",
                                "basis": "declared project script, not executed",
                                "evidence": pair(document, key, value, "toml"),
                            }
                        )
            elif path.name.startswith("requirements") and path.suffix == ".txt":
                for number, line in enumerate(lines, 1):
                    value = line.strip()
                    if not value or value.startswith(("#", "-")):
                        continue
                    match = re.match(
                        r"^([A-Za-z0-9_.-]+)(\[[A-Za-z0-9_,.-]+\])?\s*([<>=!~;].*)?$", value
                    )
                    if match:
                        dependency(
                            document,
                            "python",
                            match[1],
                            match[3] or "(unconstrained)",
                            "requirements",
                            cite(document, number),
                        )
        except (ValueError, TypeError, AttributeError, KeyError):
            diagnostics.append({"path": document.path, "code": "manifest_not_fully_parsed"})
        if path.suffix == ".py":
            try:
                tree, hits = python_rules(document.path, document.text)
            except (SyntaxError, ValueError, RecursionError):
                diagnostics.append({"path": document.path, "code": "python_rules_not_run"})
                continue
            for node in ast.walk(tree):
                if main_guard(node):
                    entrypoints.append(
                        {
                            "name": document.path,
                            "target": "__main__ guard",
                            "certainty": "inferred",
                            "basis": "source guard present, not executed",
                            "evidence": cite(document, node.lineno),
                        }
                    )
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                    for module in modules:
                        if module.split(".")[0] in {"pytest", "unittest"}:
                            tests.append(document.path)
                            test_frameworks.append(
                                {
                                    "name": module.split(".")[0],
                                    "certainty": "observation",
                                    "basis": "Python import statement, not an execution result",
                                    "evidence": cite(document, node.lineno, node.end_lineno),
                                }
                            )
            for hit in hits:
                title, severity, category, certainty, description = RULES[hit.rule_id]
                span = evidence(document, repository_id, sha, hit.start, hit.end)
                fingerprint = stable_hash(
                    [VERSION, hit.rule_id, str(document.id), hit.start, hit.end, span.quote]
                )
                findings.append(
                    FindingContract(
                        id=uuid5(NAMESPACE_URL, str(review_id) + ":" + fingerprint),
                        profile_id=profile_id,
                        review_id=review_id,
                        title=title,
                        detail=description,
                        rule_id=hit.rule_id,
                        severity=severity,
                        category=category,
                        certainty=certainty,
                        fingerprint=fingerprint,
                        evidence=[span],
                    )
                )
                if len(findings) > 200:
                    raise AnalysisLimitError("Finding budget exceeded")
    unique = {finding.fingerprint: finding for finding in findings}
    summary = {
        "scope": "verified ingested selection only; excluded files are not analyzed",
        "file_count": len(files),
        "files": files,
        "languages": dict(sorted(languages.items())),
        "major_directories": dict(sorted(directories.items())),
        "dependencies": deps,
        "framework_indicators": frameworks,
        "test_locations": sorted(set(tests)),
        "test_framework_indicators": test_frameworks,
        "ci_configurations": ci,
        "deployment_configurations": deploy,
        "documentation": {"selected_documents": sorted(docs), "count": len(docs)},
        "entrypoint_candidates": entrypoints,
        "diagnostics": diagnostics,
        "rules_version": VERSION,
        "python_rules": [
            {
                "id": key,
                "title": value[0],
                "severity": value[1],
                "category": value[2],
                "certainty": value[3],
            }
            for key, value in RULES.items()
        ],
        "limitations": [
            "Declarations do not prove installation, execution, test success or deployment.",
            (
                "Python AST rules are not a full security audit; no taint/data-flow or "
                "non-Python issue analysis."
            ),
        ],
    }
    ordered = sorted(
        unique.values(), key=lambda f: (f.evidence[0].path, f.evidence[0].line_start, f.rule_id)
    )
    if len(json.dumps(summary)) + sum(len(f.model_dump_json()) for f in ordered) > 2 * 1024 * 1024:
        raise AnalysisLimitError("Report size budget exceeded")
    return summary, ordered
