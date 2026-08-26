# © Artur Czarnecki. All rights reserved.

"""META-P0-3: repository metadata and pyproject positioning alignment contract."""

from __future__ import annotations

import json
import struct
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_METADATA_PATH = REPO_ROOT / ".github" / "repo-management" / "repository-metadata.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
WHY_INTERGRAX_PATH = REPO_ROOT / "docs" / "project" / "overview" / "WHY_INTERGRAX.md"
SOCIAL_PREVIEW_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "github" / "intergrax-social-preview.png"
CANONICAL_HOMEPAGE = (
    "https://github.com/jakbuczarnecki/intergrax/blob/main/docs/project/overview/WHY_INTERGRAX.md"
)
SOCIAL_PREVIEW_WIDTH = 1280
SOCIAL_PREVIEW_HEIGHT = 640
GITHUB_SOCIAL_PREVIEW_MAX_BYTES = 1_048_576
REPOSITORY_SOCIAL_PREVIEW_TARGET_BYTES = 1_000_000

REQUIRED_TOPIC_SUBSET = frozenset(
    {
        "ai-agents",
        "agentic-ai",
        "ai-platform",
        "ai-governance",
        "ai-safety",
        "policy-engine",
        "human-in-the-loop",
        "audit-trail",
        "durable-execution",
        "llmops",
    }
)
REMOVED_TOPICS = frozenset(
    {
        "agent-framework",
        "python",
        "tool-use",
        "ai-observability",
        "attestation",
        "traceability",
    }
)
DESCRIPTION_POSITIONING_PHRASES = (
    "operating layer",
    "governed ai applications",
    "policy",
    "authority",
    "evidence",
    "execution",
    "recovery",
)
DESCRIPTION_FORBIDDEN_FRAMING = (
    "nexus orchestration",
    "rag, memory",
    "trace/evidence, hitl",
)


@pytest.fixture(scope="module")
def repository_metadata() -> dict[str, object]:
    return json.loads(REPOSITORY_METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pyproject_project() -> dict[str, object]:
    payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return payload["project"]


def test_repository_metadata_description_positioning(repository_metadata: dict[str, object]) -> None:
    description = str(repository_metadata["description"]).lower()
    for phrase in DESCRIPTION_POSITIONING_PHRASES:
        assert phrase in description, f"description missing positioning phrase: {phrase!r}"


def test_repository_metadata_description_rejects_old_feature_list_framing(
    repository_metadata: dict[str, object],
) -> None:
    description = str(repository_metadata["description"]).lower()
    for phrase in DESCRIPTION_FORBIDDEN_FRAMING:
        assert phrase not in description, f"description reverts to old framing: {phrase!r}"


def test_repository_metadata_topics_contract(repository_metadata: dict[str, object]) -> None:
    topics = [str(topic).strip().lower() for topic in repository_metadata["topics"]]
    assert len(topics) == 20
    assert len(set(topics)) == len(topics)
    assert REQUIRED_TOPIC_SUBSET.issubset(set(topics))
    assert REMOVED_TOPICS.isdisjoint(set(topics))


def test_pyproject_description_matches_repository_metadata(
    repository_metadata: dict[str, object],
    pyproject_project: dict[str, object],
) -> None:
    assert pyproject_project["description"] == repository_metadata["description"]


def test_pyproject_keywords_match_repository_topics(
    repository_metadata: dict[str, object],
    pyproject_project: dict[str, object],
) -> None:
    repo_topics = [str(topic).strip().lower() for topic in repository_metadata["topics"]]
    pyproject_keywords = [str(keyword).strip().lower() for keyword in pyproject_project["keywords"]]
    assert pyproject_keywords == repo_topics


def test_repository_metadata_homepage_points_to_why_intergrax(
    repository_metadata: dict[str, object],
) -> None:
    assert repository_metadata["homepage"] == CANONICAL_HOMEPAGE


def test_why_intergrax_overview_exists_for_homepage() -> None:
    assert WHY_INTERGRAX_PATH.is_file(), f"missing homepage target: {WHY_INTERGRAX_PATH}"


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        assert signature == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        assert chunk_type == b"IHDR", f"missing IHDR: {path}"
        data = handle.read(length)
    width, height = struct.unpack(">II", data[:8])
    return width, height


def test_social_preview_asset_contract() -> None:
    assert SOCIAL_PREVIEW_PATH.is_file(), f"missing social preview asset: {SOCIAL_PREVIEW_PATH}"
    width, height = _png_dimensions(SOCIAL_PREVIEW_PATH)
    assert (width, height) == (SOCIAL_PREVIEW_WIDTH, SOCIAL_PREVIEW_HEIGHT)
    size_bytes = SOCIAL_PREVIEW_PATH.stat().st_size
    assert size_bytes <= GITHUB_SOCIAL_PREVIEW_MAX_BYTES
    assert size_bytes <= REPOSITORY_SOCIAL_PREVIEW_TARGET_BYTES
