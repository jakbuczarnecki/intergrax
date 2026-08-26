# © Artur Czarnecki. All rights reserved.

"""META-P0-3: repository metadata and pyproject positioning alignment contract."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_METADATA_PATH = REPO_ROOT / ".github" / "repo-management" / "repository-metadata.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

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
