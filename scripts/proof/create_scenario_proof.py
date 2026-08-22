# © Artur Czarnecki. All rights reserved.

"""Create a design-stage Scenario Proof package under platform_proofs/scenarios/."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CANONICAL_SCENARIOS_ROOT = Path("platform_proofs") / "scenarios"
SCENARIO_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN_SLUG_SEGMENTS = frozenset({".", ".."})


class ScenarioSlugError(ValueError):
    """Invalid or unsafe scenario slug."""


class ScenarioPackageExistsError(FileExistsError):
    """Target scenario package directory already exists."""


@dataclass(frozen=True, slots=True)
class ScenarioSlug:
    value: str


@dataclass(frozen=True, slots=True)
class ScenarioDesignRequest:
    slug: ScenarioSlug
    title: str
    repo_root: Path


@dataclass(frozen=True, slots=True)
class ScenarioDesignPackage:
    package_root: Path
    readme_path: Path


def resolve_repo_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().parents[2]


def validate_scenario_slug(raw_slug: str) -> ScenarioSlug:
    normalized = raw_slug.strip()
    if not normalized:
        raise ScenarioSlugError("slug must be non-empty")
    if normalized in FORBIDDEN_SLUG_SEGMENTS:
        raise ScenarioSlugError("slug must not be a path segment reserved name")
    if "/" in normalized or "\\" in normalized:
        raise ScenarioSlugError("slug must not contain path separators")
    if not SCENARIO_SLUG_PATTERN.fullmatch(normalized):
        raise ScenarioSlugError(
            "slug must be lowercase alphanumeric with underscores "
            "(e.g. ai_incident_investigation)"
        )
    return ScenarioSlug(normalized)


def scenario_package_root(repo_root: Path, slug: ScenarioSlug) -> Path:
    scenarios_root = (repo_root / CANONICAL_SCENARIOS_ROOT).resolve()
    package_root = (scenarios_root / slug.value).resolve()
    try:
        package_root.relative_to(scenarios_root)
    except ValueError:
        raise ScenarioSlugError("slug resolves outside platform_proofs/scenarios/")
    return package_root


def build_design_readme(title: str) -> str:
    return (
        f"# {title}\n\n"
        "## Scenario identity\n\n"
        f"- **Title:** {title}\n"
        "- **Public question:** _(design-stage — to be qualified)_\n"
        "- **Lifecycle status:** DESIGN / NOT YET ACCEPTED\n"
        "- **Executable proof:** No executable proof, evidence, or report exists yet. "
        "This package is a scenario design document awaiting human Scenario Quality Gate.\n\n"
        "## A. SCENARIO\n\n"
        "### Real problem\n\n"
        "_To be qualified._\n\n"
        "### Who has the problem\n\n"
        "_To be qualified._\n\n"
        "### Why it matters\n\n"
        "_To be qualified._\n\n"
        "### Failure consequences\n\n"
        "_To be qualified._\n\n"
        "### Why it is difficult\n\n"
        "_To be qualified._\n\n"
        "### Naive / simple failure mode\n\n"
        "_To be qualified._\n\n"
        "### WOW factor\n\n"
        "_To be qualified._\n\n"
        "### Skeptic Challenge\n\n"
        "_To be qualified._\n\n"
        "### Adversarial conditions\n\n"
        "_To be qualified._\n\n"
        "### Scenario Quality Gate\n\n"
        "_To be qualified._\n\n"
        "## B. SOLUTION\n\n"
        "### Desired behavior\n\n"
        "_To be qualified._\n\n"
        "### Step-by-step story\n\n"
        "_To be qualified._\n\n"
        "### Guarantees\n\n"
        "_To be qualified._\n\n"
        "### Claim\n\n"
        "_To be qualified._\n\n"
        "### PASS\n\n"
        "_To be qualified._\n\n"
        "### FAIL\n\n"
        "_To be qualified._\n\n"
        "### Adversarial attacks\n\n"
        "_To be qualified._\n\n"
        "### Excluded claims\n\n"
        "_To be qualified._\n\n"
        "### Limitations\n\n"
        "_To be qualified._\n\n"
        "## C. INTERGRAX FIT\n\n"
        "NOT YET PERFORMED\n\n"
        "## D. GAP DECISION\n\n"
        "NOT YET PERFORMED\n\n"
        "## E. PROOF BUILD\n\n"
        "NOT STARTED — blocked on scenario acceptance and capability-fit.\n"
    )


DESIGN_STAGE_REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Scenario identity",
    "## A. SCENARIO",
    "### Real problem",
    "### Who has the problem",
    "### Why it matters",
    "### Failure consequences",
    "### Why it is difficult",
    "### Naive / simple failure mode",
    "### WOW factor",
    "### Skeptic Challenge",
    "### Adversarial conditions",
    "### Scenario Quality Gate",
    "## B. SOLUTION",
    "### Desired behavior",
    "### Step-by-step story",
    "### Guarantees",
    "### Claim",
    "### PASS",
    "### FAIL",
    "### Adversarial attacks",
    "### Excluded claims",
    "### Limitations",
    "## C. INTERGRAX FIT",
    "## D. GAP DECISION",
    "## E. PROOF BUILD",
    "DESIGN / NOT YET ACCEPTED",
    "NOT YET PERFORMED",
    "NOT STARTED — blocked on scenario acceptance and capability-fit.",
)

DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        "proof.json",
        "run_proof.py",
        "evaluator.py",
        "evidence_builder.py",
        "scenario.py",
        "scenarios.py",
        "docker-compose.yml",
        ".env.example",
        "evidence.json",
        "proof-result.json",
        "report.html",
    }
)


def create_scenario_design_package(request: ScenarioDesignRequest) -> ScenarioDesignPackage:
    scenarios_root = (request.repo_root / CANONICAL_SCENARIOS_ROOT).resolve()
    scenarios_root.mkdir(parents=True, exist_ok=True)
    package_root = scenario_package_root(request.repo_root, request.slug)
    if package_root.exists():
        raise ScenarioPackageExistsError(
            f"scenario package already exists: {package_root}"
        )

    package_root.mkdir(parents=False, exist_ok=False)
    readme_path = package_root / "README.md"
    readme_path.write_text(build_design_readme(request.title), encoding="utf-8")
    return ScenarioDesignPackage(package_root=package_root, readme_path=readme_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a design-stage Scenario Proof package.",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Filesystem-safe scenario slug (lowercase, underscores).",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Human-readable scenario title.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (defaults to parent of scripts/proof/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    slug = validate_scenario_slug(args.slug)
    title = args.title.strip()
    if not title:
        print("title must be non-empty", file=sys.stderr)
        return 2

    try:
        package = create_scenario_design_package(
            ScenarioDesignRequest(slug=slug, title=title, repo_root=repo_root),
        )
    except ScenarioSlugError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ScenarioPackageExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(package.package_root.relative_to(repo_root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
