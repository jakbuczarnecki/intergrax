#!/usr/bin/env python3
"""DEP-4 dependency ownership and version-policy gate."""

from __future__ import annotations

import argparse
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = REPO_ROOT / "pyproject.toml"

# This is deliberately an explicit review surface. Adding a core dependency
# requires changing this map and the qualification evidence in the same PR.
CORE_ALLOWLIST: dict[str, str] = {
    "fastapi": "CORE_SERVER",
    "uvicorn": "CORE_SERVER",
    "starlette": "CORE_SERVER",
    "httpx": "CORE_FOUNDATION",
    "python-multipart": "CORE_SERVER",
    "pydantic": "CORE_FOUNDATION",
    "cryptography": "CORE_FOUNDATION",
    "python-dotenv": "CORE_FOUNDATION",
    "pyyaml": "CORE_FOUNDATION",
    "tqdm": "CORE_FOUNDATION",
    "openai": "CORE_FOUNDATION",
    "tiktoken": "CORE_FOUNDATION",
    "boto3": "CORE_SERVER",
    "numpy": "CORE_FOUNDATION",
    "chardet": "CORE_FOUNDATION",
}

FORBIDDEN_CORE_NAMES = frozenset(
    {
        "torch",
        "sentence-transformers",
        "transformers",
        "openai-whisper",
        "chromadb",
        "qdrant-client",
        "pinecone",
        "streamlit",
        "fastmcp",
        "mcp",
        "anthropic",
        "mistralai",
        "ollama",
        "google-genai",
        "cohere",
        "beautifulsoup4",
        "trafilatura",
        "python-docx",
        "openpyxl",
        "xlrd",
        "pytesseract",
        "pillow",
        "pymupdf",
        "yt-dlp",
        "webvtt-py",
        "opencv-python-headless",
        "pandas",
    }
)

TRANSITIVE_ONLY_NAMES = frozenset({"mcp"})

# These are intentional shared ownership boundaries, not accidental duplicate
# declarations. Other core/extra overlaps fail closed.
INTENTIONAL_CORE_EXTRA_SHARING = frozenset(
    {
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "pyyaml",
        "openai",
        "tiktoken",
        "boto3",
    }
)

# A bound is required for every occurrence of these high-risk direct packages.
# The expected mode is a minimum policy: an exact pin also satisfies a bounded
# policy, but the project avoids exact pins unless qualification requires one.
HIGH_RISK_POLICY: dict[str, str] = {
    "fastapi": "BOUNDED_MAJOR",
    "uvicorn": "BOUNDED_MAJOR",
    "starlette": "BOUNDED_MAJOR",
    "httpx": "BOUNDED_MAJOR",
    "pydantic": "BOUNDED_MAJOR",
    "cryptography": "BOUNDED_MAJOR",
    "numpy": "EXACT_PIN",
    "pandas": "BOUNDED_MAJOR",
    "openai": "BOUNDED_MAJOR",
    "boto3": "BOUNDED_MAJOR",
    "tiktoken": "BOUNDED_MAJOR",
    "anthropic": "BOUNDED_MAJOR",
    "mistralai": "BOUNDED_MAJOR",
    "ollama": "BOUNDED_MAJOR",
    "google-genai": "BOUNDED_MAJOR",
    "cohere": "BOUNDED_MAJOR",
    "chromadb": "EXACT_PIN",
    "qdrant-client": "BOUNDED_MAJOR",
    "pinecone": "BOUNDED_MAJOR",
    "fastmcp": "BOUNDED_MAJOR",
    "openai-whisper": "BOUNDED_MAJOR",
    "langchain-community": "BOUNDED_MAJOR",
    "langchain-text-splitters": "BOUNDED_MAJOR",
    "langchain-ollama": "BOUNDED_MAJOR",
    "langgraph": "BOUNDED_MAJOR",
}


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _project_data(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)["project"]


def _parse_requirements(
    project: dict[str, Any],
) -> tuple[list[Requirement], dict[str, list[Requirement]], list[str]]:
    problems: list[str] = []
    raw_core = project.get("dependencies", [])
    raw_extras = project.get("optional-dependencies", {})
    if not isinstance(raw_core, list) or not isinstance(raw_extras, dict):
        return [], {}, ["MALFORMED_PROJECT: dependencies and optional-dependencies must be valid"]

    def parse(raw: object, owner: str) -> list[Requirement]:
        if not isinstance(raw, list):
            problems.append(f"MALFORMED_EXTRA: {owner} must be a list")
            return []
        result: list[Requirement] = []
        for value in raw:
            if not isinstance(value, str):
                problems.append(f"MALFORMED_REQUIREMENT: {owner}: {value!r}")
                continue
            try:
                result.append(Requirement(value))
            except InvalidRequirement as exc:
                problems.append(f"INVALID_REQUIREMENT: {owner}: {value!r}: {exc}")
        return result

    core = parse(raw_core, "core")
    extras = {str(name): parse(values, str(name)) for name, values in raw_extras.items()}
    return core, extras, problems


def classify_version_policy(requirement: Requirement) -> str:
    specifiers = list(requirement.specifier)
    if len(specifiers) == 1 and specifiers[0].operator in {"==", "==="}:
        return "EXACT_PIN"
    if any(specifier.operator in {"<", "<=", "~="} for specifier in specifiers):
        return "BOUNDED_MAJOR"
    return "UNBOUNDED_MAJOR"


def check_project(path: Path = DEFAULT_PROJECT) -> list[str]:
    project = _project_data(path)
    core, extras, problems = _parse_requirements(project)
    core_names = [normalize_name(requirement.name) for requirement in core]
    all_requirements = [("core", requirement) for requirement in core]
    all_requirements.extend(
        (extra, requirement)
        for extra, requirements in extras.items()
        for requirement in requirements
    )

    for name, count in Counter(core_names).items():
        if count > 1:
            problems.append(f"DUPLICATE_CORE_DECLARATION: {name}")

    for requirement in core:
        name = normalize_name(requirement.name)
        if name not in CORE_ALLOWLIST:
            problems.append(f"CORE_OWNERSHIP_VIOLATION: {name}")
        if name in FORBIDDEN_CORE_NAMES:
            problems.append(f"FORBIDDEN_CORE_PACKAGE: {name}")
        if name.startswith("langchain") or name == "langgraph":
            problems.append(f"LANGCHAIN_CORE_LEAK: {name}")

    extras_by_name: dict[str, set[str]] = {}
    for extra, requirement in all_requirements:
        name = normalize_name(requirement.name)
        extras_by_name.setdefault(name, set()).add(extra)
        if name in TRANSITIVE_ONLY_NAMES:
            problems.append(f"PROHIBITED_DIRECT_TRANSITIVE: {extra}: {name}")

        expected = HIGH_RISK_POLICY.get(name)
        if expected is not None:
            actual = classify_version_policy(requirement)
            if actual == "UNBOUNDED_MAJOR":
                problems.append(f"UNBOUNDED_MAJOR: {extra}: {requirement}")
            elif expected == "EXACT_PIN" and actual != "EXACT_PIN":
                problems.append(
                    f"EXACT_PIN_REQUIRED: {extra}: {requirement} (actual {actual})"
                )

    for name, owners in extras_by_name.items():
        if "core" in owners and len(owners) > 1 and name not in INTENTIONAL_CORE_EXTRA_SHARING:
            problems.append(
                f"DUPLICATE_CORE_EXTRA_OWNERSHIP: {name}: {', '.join(sorted(owners))}"
            )

    llm_all = {normalize_name(requirement.name) for requirement in extras.get("llm-all", [])}
    for name in sorted(llm_all):
        if name.startswith("langchain") or name.startswith("langgraph"):
            problems.append(f"LLM_ALL_LANGCHAIN_LEAK: {name}")

    return sorted(set(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "project",
        nargs="?",
        type=Path,
        default=DEFAULT_PROJECT,
        help="pyproject.toml path (default: repository pyproject.toml)",
    )
    args = parser.parse_args()
    problems = check_project(args.project.resolve())
    if problems:
        print("dependency governance: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1

    project = _project_data(args.project.resolve())
    core_count = len(project.get("dependencies", []))
    extra_count = sum(len(values) for values in project.get("optional-dependencies", {}).values())
    print("dependency governance: OK")
    print(f"core direct dependencies: {core_count}")
    print(f"optional direct declarations: {extra_count}")
    print("core ownership, optional-family, duplicate, and major-bound checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
