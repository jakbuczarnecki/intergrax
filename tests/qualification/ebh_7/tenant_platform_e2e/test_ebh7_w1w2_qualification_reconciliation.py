# © Artur Czarnecki. All rights reserved.

"""EBH-7 W1/W2 qualification reconciliation tests (PARTIAL — not disposition EXACT)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_w1w2_reconciliation_probes import (
    probe_governed_root_launch_denied_before_intake,
    probe_plugin_host_activation_cross_tenant_open,
    probe_relational_store_sqlite_duckdb_dual_slug,
    probe_resolve_from_profile_sqlite_sanctioned,
    probe_resolve_from_profile_unregistered_fail_closed,
    probe_tool_runtime_invoke_chain_cross_tenant_denied,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


@pytest.mark.asyncio
async def test_ebh7_w1w2_ten03_governed_root_launch_admission_denies_before_intake() -> None:
    await probe_governed_root_launch_denied_before_intake()


@pytest.mark.asyncio
async def test_ebh7_w1w2_ten04_tool_runtime_invoke_reaches_invoker_tenant_guard() -> None:
    await probe_tool_runtime_invoke_chain_cross_tenant_denied()


def test_ebh7_w1w2_ten05_resolve_from_profile_sqlite_sanctioned(tmp_path: Path) -> None:
    probe_resolve_from_profile_sqlite_sanctioned(tmp_path)


def test_ebh7_w1w2_ten05_resolve_from_profile_unregistered_fail_closed() -> None:
    probe_resolve_from_profile_unregistered_fail_closed()


def test_ebh7_w1w2_ten13_plugin_host_cross_tenant_activation_blocker() -> None:
    probe_plugin_host_activation_cross_tenant_open()


def test_ebh7_w1w2_rpl_relational_store_sqlite_duckdb_dual_slug(tmp_path: Path) -> None:
    probe_relational_store_sqlite_duckdb_dual_slug(tmp_path)
