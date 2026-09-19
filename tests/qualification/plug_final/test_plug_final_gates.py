# © Artur Czarnecki. All rights reserved.

"""PLUG-FINAL — enterprise closure gates (catalog integrity + cross-domain checks)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from intergrax.applications.contracts.platform_plugin_evidence import ApplicationPlatformPluginEvidence
from intergrax.core.plugins.admission import DomainPluginLoadReport
from tests.qualification.plug_03.catalog import PLUG_03_SURFACE_MATRIX
from tests.qualification.plug_final.catalog import (
    PLUG_FINAL_MAPPED_NODE_IDS,
    PLUG_FINAL_Q4_PUBLIC_SURFACES,
    PLUG_FINAL_SURFACE_MATRIX,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_REQUIRED_INVENTORY_SURFACES = frozenset(
    {
        "Tools / ToolPlugin",
        "ToolInvocationPattern",
        "Skills / SkillPlugin",
        "Integrations",
        "Memory — UserProfileStore",
        "Memory — SessionStorage",
        "RAG — chunker",
        "RAG — retriever",
        "RAG — reranker",
        "RAG — embedding",
        "RAG — document handler",
        "SecurityDefensePlugin",
        "PolicyRuleHandler",
        "Vendor Knowledge provider",
        "RuntimePlugin",
        "Agents",
        "Token optimization",
        "Decision — strategies",
    }
)

_Q4_REQUIRED_EVIDENCE_KINDS = frozenset(
    {
        "CANONICAL_CONSUMPTION",
        "SELECTION",
        "DEFAULT_BYPASS",
        "FAIL_CLOSED",
        "ADMISSION",
        "DISCOVERY",
        "EVIDENCE",
        "GATE",
    }
)


def test_plug_final_matrix_covers_inventory_surfaces() -> None:
    surfaces = {row.surface for row in PLUG_FINAL_SURFACE_MATRIX}
    missing = _REQUIRED_INVENTORY_SURFACES - surfaces
    assert not missing, f"PLUG-FINAL matrix missing surfaces: {sorted(missing)}"


def test_plug_final_q4_public_rows_have_evidence() -> None:
    for row in PLUG_FINAL_SURFACE_MATRIX:
        if row.final_q != "Q4" or row.classification != "PUBLIC_EXTERNAL_PLUGIN":
            continue
        assert row.evidence, f"Q4 public row {row.surface!r} must list evidence node ids"
        kinds = frozenset(kind for ref in row.evidence for kind in ref.kinds)
        assert "CANONICAL_CONSUMPTION" in kinds or "EVIDENCE" in kinds, (
            f"Q4 row {row.surface!r} needs CANONICAL_CONSUMPTION or EVIDENCE proof"
        )


def test_plug_final_q4_surfaces_align_with_plug_03_pass_rows() -> None:
    plug03_q4_public = {
        row.surface
        for row in PLUG_03_SURFACE_MATRIX
        if row.level == "Q4" and row.classification == "PUBLIC_EXTERNAL_PLUGIN" and row.status == "PASS"
    }
    # PLUG-03 groups context rows; PLUG-FINAL uses one combined context row.
    plug03_normalized = plug03_q4_public - {
        "Context — token counter",
        "Context — budget / compaction / degradation",
    } | {"Context — token / budget / compaction / degradation"}
    final_q4 = PLUG_FINAL_Q4_PUBLIC_SURFACES
    assert plug03_normalized <= final_q4, f"PLUG-03 Q4 not reflected in PLUG-FINAL: {plug03_normalized - final_q4}"


def test_application_platform_plugin_evidence_is_immutable_metadata() -> None:
    assert is_dataclass(ApplicationPlatformPluginEvidence)
    field_names = {f.name for f in fields(ApplicationPlatformPluginEvidence)}
    assert field_names == {"_domain_reports"}

    empty = DomainPluginLoadReport.empty("intergrax.tools")
    evidence = ApplicationPlatformPluginEvidence.from_domain_reports({"tools": empty})
    reports = evidence.domain_reports
    assert isinstance(reports, MappingProxyType)
    with pytest.raises(TypeError):
        reports["extra"] = empty  # type: ignore[index]


def test_plug_final_catalog_evidence_nodes_collect() -> None:
    node_ids = [nid for nid in PLUG_FINAL_MAPPED_NODE_IDS if nid]
    assert node_ids
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
