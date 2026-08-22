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

# Canonical lifecycle wording — see PLATFORM_PROOF_AUTHORING_GUIDE § Scenario README standard.
LIFECYCLE_DESIGN_NOT_ACCEPTED = "DESIGN / NOT YET ACCEPTED"
LIFECYCLE_ACCEPTED_FOR_IMPLEMENTATION = "ACCEPTED FOR IMPLEMENTATION"


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


VISUAL_STORY_AUTHORING_HINT = (
    "<!-- Add scenario-owned explanatory visual after Scenario Quality Gate.\n"
    "     Use light/dark SVG per docs/project/technical/guides/DOCUMENTATION_DESIGN_SYSTEM.md.\n"
    "     Do not use decorative imagery or fake execution results. -->"
)


def build_design_readme(title: str) -> str:
    return (
        f"# {title}\n\n"
        "> **_(Public question — qualify before Scenario Quality Gate)_**\n\n"
        "> _(One- or two-sentence public explanation — qualify.)_\n\n"
        "> [!NOTE]\n"
        f"> **Scenario status:** {LIFECYCLE_DESIGN_NOT_ACCEPTED} — "
        "awaiting human Scenario Quality Gate; no executable proof, evidence, or report exists yet.\n\n"
        "## At a glance\n\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        "| **Problem** | _(qualify)_ |\n"
        "| **Observed impact** | _(qualify)_ |\n"
        "| **Trap** | _(qualify)_ |\n"
        "| **Decision risk** | _(qualify)_ |\n"
        "| **Scenario outcome** | RESOLVED or UNRESOLVED |\n"
        f"| **Status** | {LIFECYCLE_DESIGN_NOT_ACCEPTED} |\n"
        "| **Proof class** | SCENARIO |\n\n"
        "## Visual proof story\n\n"
        f"{VISUAL_STORY_AUTHORING_HINT}\n\n"
        "_Visual placeholder — enrich after Scenario Quality Gate._\n\n"
        "## The real problem\n\n"
        "_Brief public summary — expand in § A. SCENARIO._\n\n"
        "## The trap\n\n"
        "_What the naive answer gets wrong — expand in § A. SCENARIO._\n\n"
        "## What this proof attempts to guarantee\n\n"
        "_Bounded claim summary — expand in § B. SOLUTION._\n\n"
        "## Scenario outcomes\n\n"
        "_RESOLVED and UNRESOLVED semantics — expand in § B. SOLUTION._\n\n"
        "## PASS / FAIL (summary)\n\n"
        "| PASS | FAIL |\n"
        "| --- | --- |\n"
        "| _(qualify)_ | _(qualify)_ |\n\n"
        "_Full normative contract in § B. SOLUTION._\n\n"
        "---\n\n"
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
        "### Conditional authoring prompts _(complete when relevant)_\n\n"
        "**Hidden truth / evaluator leakage:** Does this scenario have hidden fixture truth "
        "or expected behavior? If yes, how is it isolated from model-visible context?\n\n"
        "**Evidence boundary:** What is legitimately observable by the system?\n\n"
        "**Alternative hypotheses / failure alternatives:** What plausible alternatives must "
        "the system distinguish?\n\n"
        "**Independence:** If any verifier/reviewer/critic is called independent, what exactly "
        "makes it independent?\n\n"
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
        "NOT STARTED — blocked on scenario acceptance and capability-fit.\n\n"
        "## Latest verified run\n\n"
        "> [!NOTE]\n"
        "> **Not yet available.** Populated only after a real proof run and report acceptance.\n\n"
        "## Report / evidence / source / run\n\n"
        "> [!NOTE]\n"
        "> **Not yet available.** Links appear here after implementation and execution.\n"
    )


DESIGN_STAGE_REQUIRED_SECTIONS: tuple[str, ...] = (
    "## At a glance",
    "## Visual proof story",
    "## The real problem",
    "## The trap",
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
    "### Conditional authoring prompts",
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
    "## Latest verified run",
    LIFECYCLE_DESIGN_NOT_ACCEPTED,
    "NOT YET PERFORMED",
    "NOT STARTED — blocked on scenario acceptance and capability-fit.",
    VISUAL_STORY_AUTHORING_HINT.split("\n", maxsplit=1)[0],
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
