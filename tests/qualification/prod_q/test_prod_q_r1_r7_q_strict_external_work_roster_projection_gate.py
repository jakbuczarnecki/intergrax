# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q — fail-closed gates for revision-bound external-contractor roster."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_contractor_application.host.strict_sanctioned_external_work_registry_projection import (
    _external_contractor_factory_keys,
)
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.runtime_agent_factory_resolver import _factory_key
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _DIGEST_A,
    _EXTERNAL_CONTRACTOR_FACTORY_REF,
    _ac3_revision_bound_strict_external_work_host_manifest,
    _build_echo_artifact,
    _revision,
    _roster,
    _roster_with_strict_external_work,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_ROSTER_ERROR = "revision-bound 'external_contractor_adapter' roster entry"


def _bundle(
    *,
    roster,
    manifest: ApplicationManifest,
    tmp_path: Path,
    marker: str,
) -> RegistryProjectionInputBundle:
    _artifact_root, digest, lock = _build_echo_artifact(
        tmp_path,
        marker=marker,
        revision_bound_strict_external_work=True,
    )
    revision = _revision(
        "rev-roster-gate",
        roster=roster,
        artifact_digest=digest,
        lock=lock,
    )
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        factory_resolver=None,
        builders=None,
        materialization_artifact_digest=digest,
        roster_contract_authority=None,
    )


def test_missing_external_contractor_roster_entry_fail_closed(tmp_path: Path) -> None:
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="search-only",
        agents=[AgentBinding(contract_id="search")],
    )
    with pytest.raises(ValueError, match=_ROSTER_ERROR):
        _external_contractor_factory_keys(
            _bundle(
                roster=_roster(),
                manifest=manifest,
                tmp_path=tmp_path,
                marker="missing-external",
            ),
        )


def test_disabled_external_contractor_entry_fail_closed(tmp_path: Path) -> None:
    roster = _roster_with_strict_external_work()
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(update={"effective_enablement": False})
    roster = roster.model_copy(update={"entries": tuple(entries)})
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    with pytest.raises(ValueError, match=_ROSTER_ERROR):
        _external_contractor_factory_keys(
            _bundle(
                roster=roster,
                manifest=manifest,
                tmp_path=tmp_path,
                marker="disabled-external",
            ),
        )


def test_incorrect_contract_id_fail_closed(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    bad_manifest = manifest.model_copy(
        update={
            "agents": [
                manifest.agents[0],
                manifest.agents[1].model_copy(
                    update={"contract_id": "not_external_contractor_adapter"},
                ),
            ],
        },
    )
    roster = _roster_with_strict_external_work()
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(
        update={
            "logical_agent_id": "not_external_contractor_adapter",
            "manifest_origin_ref": "manifest:agents/not_external_contractor_adapter",
        },
    )
    roster = roster.model_copy(update={"entries": tuple(entries)})
    with pytest.raises(ValueError, match=_ROSTER_ERROR):
        _external_contractor_factory_keys(
            _bundle(
                roster=roster,
                manifest=bad_manifest,
                tmp_path=tmp_path,
                marker="wrong-contract-id",
            ),
        )


def test_unknown_factory_reference_still_emits_key_when_contract_matches(
    tmp_path: Path,
) -> None:
    roster = _roster_with_strict_external_work()
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(
        update={
            "factory_reference": AgentBindingFactoryReference(
                factory_path="missing.factory.build_agent",
            ),
        },
    )
    roster = roster.model_copy(update={"entries": tuple(entries)})
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    keys = _external_contractor_factory_keys(
        _bundle(
            roster=roster,
            manifest=manifest,
            tmp_path=tmp_path,
            marker="unknown-factory-ref",
        ),
    )
    assert keys == frozenset(
        {
            _factory_key(
                _DIGEST_A,
                AgentBindingFactoryReference(factory_path="missing.factory.build_agent"),
            ),
        },
    )


def test_positive_revision_bound_factory_key(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    keys = _external_contractor_factory_keys(
        _bundle(
            roster=roster,
            manifest=manifest,
            tmp_path=tmp_path,
            marker="positive-key",
        ),
    )
    assert keys == frozenset({_factory_key(_DIGEST_A, _EXTERNAL_CONTRACTOR_FACTORY_REF)})


def test_incorrect_package_digest_changes_canonical_factory_key(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    canonical = _external_contractor_factory_keys(
        _bundle(
            roster=roster,
            manifest=manifest,
            tmp_path=tmp_path,
            marker="digest-canonical",
        ),
    )
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(update={"package_digest": "sha256:" + ("9" * 64)})
    roster = roster.model_copy(update={"entries": tuple(entries)})
    mutated = _external_contractor_factory_keys(
        _bundle(
            roster=roster,
            manifest=manifest,
            tmp_path=tmp_path,
            marker="digest-mutated",
        ),
    )
    assert canonical != mutated


def test_runtime_revision_id_does_not_change_factory_keys(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    bundle = _bundle(
        roster=roster,
        manifest=manifest,
        tmp_path=tmp_path,
        marker="stale-revision",
    )
    stale = RegistryProjectionInputBundle(
        runtime_revision=bundle.runtime_revision.model_copy(
            update={"runtime_revision_id": "rev-stale"},
        ),
        effective_roster=bundle.effective_roster,
        manifest=bundle.manifest,
        build_context=bundle.build_context,
        factory_resolver=bundle.factory_resolver,
        builders=bundle.builders,
        materialization_artifact_digest=bundle.materialization_artifact_digest,
        roster_contract_authority=bundle.roster_contract_authority,
    )
    assert _external_contractor_factory_keys(bundle) == _external_contractor_factory_keys(stale)


def test_cross_environment_roster_does_not_remove_external_contractor_key(
    tmp_path: Path,
) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work().model_copy(
        update={"application_environment_id": "env-other-tenant"},
    )
    keys = _external_contractor_factory_keys(
        _bundle(
            roster=roster,
            manifest=manifest,
            tmp_path=tmp_path,
            marker="cross-env",
        ),
    )
    assert keys == frozenset({_factory_key(_DIGEST_A, _EXTERNAL_CONTRACTOR_FACTORY_REF)})
