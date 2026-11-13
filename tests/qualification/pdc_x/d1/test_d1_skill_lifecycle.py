# © Artur Czarnecki. All rights reserved.

"""PDC-D1-R018 — skill disable / remove / upgrade on the host lifecycle authority.

Skill lifecycle stays separate from agent lifecycle (SKILL-X F-04): these operations act on
``SkillHostLifecycleService`` only and never touch distribution-plane agent installations.
"""

from __future__ import annotations

import pytest
from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffDisposition
from intergrax.skills.catalog import SkillCatalogEntry, SkillPackageResolution
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.host_lifecycle import (
    SKILL_EVENT_DISABLED,
    SKILL_EVENT_REMOVED,
    SKILL_EVENT_UPGRADED,
    SkillHostLifecycleService,
    SkillLifecycleOutcome,
)
from intergrax.skills.identity import SkillPackageCandidate
from intergrax.skills.registry.profile import is_skill_enabled
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.runtime import SkillRegistry
from testing_support.canonical_me15_reference_skill import (
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V2,
)
from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer
from testing_support.me15_skill_catalog_provider import (
    ME15_CATALOG_ENTRY_ID,
    Me15SkillCatalogProvider,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_HOST = "host-d1"
_SOURCE = "official.intergrax.d1"
_SKILL = "skills.d1.search"
_PACKAGE_REF = "pkg:skill/d1-search"


def _digest(char: str) -> str:
    return "sha256:" + (char * 64)


def _resolution(
    version: str,
    digest_char: str,
    *,
    package_reference: str = _PACKAGE_REF,
) -> SkillPackageResolution:
    return SkillPackageResolution(
        entry=SkillCatalogEntry(
            catalog_entry_id="entry-d1",
            catalog_source_id=_SOURCE,
            logical_skill_id=_SKILL,
            package_reference=package_reference,
            display_name="D1 search skill",
        ),
        package_candidate=SkillPackageCandidate(
            logical_skill_id=_SKILL,
            package_reference=package_reference,
            package_version=version,
            package_digest=_digest(digest_char),
        ),
    )


class _Materializer:
    """Registers the resolved release into the host registry (same contract as production)."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        fail: bool = False,
        wrong_binding: bool = False,
    ) -> None:
        self._registry = registry
        self._fail = fail
        self._wrong_binding = wrong_binding

    def materialize(self, resolution: SkillPackageResolution) -> tuple[str, SkillRuntimeBindingMetadata]:
        candidate = resolution.package_candidate
        assert candidate.package_digest is not None
        binding = SkillRuntimeBindingMetadata(
            catalog_source_id=_SOURCE,
            logical_skill_id=candidate.logical_skill_id,
            package_reference=candidate.package_reference,
            version_label=candidate.package_version,
            content_digest=(_digest("f") if self._wrong_binding else candidate.package_digest),
        )
        self._registry.register(
            SkillManifest(
                skill_id=candidate.logical_skill_id,
                version=candidate.package_version,
                description=f"d1 skill {candidate.package_version}",
            ),
            binding=binding,
        )
        if self._fail:
            raise RuntimeError("simulated materialization failure")
        return candidate.logical_skill_id, binding


def _installed(*, enabled: bool = True) -> SkillHostLifecycleService:
    lifecycle = SkillHostLifecycleService(host_profile_id=_HOST)
    ack = lifecycle.bind(
        operation_id="op-install",
        host_profile_id=_HOST,
        resolved=_resolution("1.0.0", "a"),
        materializer=_Materializer(lifecycle.registry),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.ACCEPTED
    if enabled:
        lifecycle.enable_installed_skill(_SKILL)
    return lifecycle


def _upgrade(
    lifecycle: SkillHostLifecycleService,
    resolution: SkillPackageResolution,
    *,
    operation_id: str = "op-upgrade",
    materializer: object | None = None,
    host_profile_id: str = _HOST,
):
    return lifecycle.upgrade_skill(
        operation_id=operation_id,
        host_profile_id=host_profile_id,
        resolved=resolution,
        materializer=materializer or _Materializer(lifecycle.registry),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- disable


def test_r018_disable_withdraws_enablement_but_keeps_the_skill_installed() -> None:
    lifecycle = _installed()
    assert is_skill_enabled(lifecycle.skill_profile, _SKILL)

    result = lifecycle.disable_skill(_SKILL)

    assert result.outcome is SkillLifecycleOutcome.APPLIED
    assert result.event_type == SKILL_EVENT_DISABLED
    assert lifecycle.is_bound(_SKILL)
    assert not is_skill_enabled(lifecycle.skill_profile, _SKILL)
    binding = lifecycle.binding_metadata(_SKILL)
    assert binding is not None and binding.version_label == "1.0.0"  # release identity retained

    again = lifecycle.disable_skill(_SKILL)
    assert again.outcome is SkillLifecycleOutcome.NOOP and again.event_type is None

    lifecycle.enable_installed_skill(_SKILL)  # explicit re-enablement still possible
    assert is_skill_enabled(lifecycle.skill_profile, _SKILL)


def test_r018_disable_of_an_unknown_skill_is_rejected_without_event() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id=_HOST)
    result = lifecycle.disable_skill("skills.never.installed")
    assert result.outcome is SkillLifecycleOutcome.REJECTED
    assert result.event_type is None


def test_r018_disable_reports_bundle_enablement_instead_of_pretending() -> None:
    lifecycle = _installed(enabled=False)
    lifecycle.skill_profile = lifecycle.skill_profile.model_copy(
        update={"enabled_bundles": ["unknown-bundle"], "enabled": []}
    )
    result = lifecycle.disable_skill(_SKILL)
    assert result.outcome in {SkillLifecycleOutcome.NOOP, SkillLifecycleOutcome.REJECTED}
    assert result.event_type is None


# --------------------------------------------------------------------------- remove


def test_r018_remove_uninstalls_skill_and_fails_closed_on_old_handoff_replay() -> None:
    lifecycle = _installed()

    removed = lifecycle.remove_skill(_SKILL)

    assert removed.outcome is SkillLifecycleOutcome.APPLIED
    assert removed.event_type == SKILL_EVENT_REMOVED
    assert not lifecycle.is_bound(_SKILL)
    assert lifecycle.binding_metadata(_SKILL) is None
    assert _SKILL not in lifecycle.skill_profile.enabled
    assert lifecycle.remove_skill(_SKILL).outcome is SkillLifecycleOutcome.NOOP

    replay = lifecycle.bind(
        operation_id="op-install",  # the pre-removal handoff must not resurrect the skill
        host_profile_id=_HOST,
        resolved=_resolution("1.0.0", "a"),
        materializer=_Materializer(lifecycle.registry),
    )
    assert replay.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert not lifecycle.is_bound(_SKILL)

    reinstall = lifecycle.bind(
        operation_id="op-reinstall",
        host_profile_id=_HOST,
        resolved=_resolution("1.0.0", "a"),
        materializer=_Materializer(lifecycle.registry),
    )
    assert reinstall.disposition is DomainLifecycleHandoffDisposition.ACCEPTED
    assert lifecycle.is_bound(_SKILL)
    assert not is_skill_enabled(lifecycle.skill_profile, _SKILL)  # removal reset enablement


def test_r018_remove_is_refused_while_another_installed_skill_requires_it() -> None:
    lifecycle = _installed()
    lifecycle.registry.register(
        SkillManifest(
            skill_id="skills.d1.dependent",
            description="needs search",
            requires_skills=(_SKILL,),
        )
    )

    result = lifecycle.remove_skill(_SKILL)

    assert result.outcome is SkillLifecycleOutcome.REJECTED
    assert "skills.d1.dependent" in result.reason_detail
    assert result.event_type is None
    assert lifecycle.is_bound(_SKILL)

    lifecycle.remove_skill("skills.d1.dependent")
    assert lifecycle.remove_skill(_SKILL).outcome is SkillLifecycleOutcome.APPLIED


# --------------------------------------------------------------------------- upgrade


def test_r018_upgrade_replaces_the_release_and_preserves_enablement() -> None:
    lifecycle = _installed()

    result = _upgrade(lifecycle, _resolution("1.1.0", "b"))

    assert result.outcome is SkillLifecycleOutcome.APPLIED
    assert result.event_type == SKILL_EVENT_UPGRADED
    assert result.domain_reference == f"skill:{_SKILL}@1.1.0:{_digest('b')}"
    binding = lifecycle.binding_metadata(_SKILL)
    assert binding is not None
    assert (binding.version_label, binding.content_digest) == ("1.1.0", _digest("b"))
    assert lifecycle.registry.get(_SKILL).manifest.version == "1.1.0"
    assert is_skill_enabled(lifecycle.skill_profile, _SKILL)

    replay = _upgrade(lifecycle, _resolution("1.1.0", "b"))
    assert replay.outcome is SkillLifecycleOutcome.NOOP and replay.event_type is None


@pytest.mark.parametrize(
    ("resolution", "fragment"),
    [
        (_resolution("1.0.0", "c"), "not newer"),  # same version, new digest
        (_resolution("0.9.0", "c"), "major version"),  # cross-major downgrade
        (_resolution("2.0.0", "c"), "major version"),  # major change is not in-place
        (_resolution("1.0.0", "a"), "already installed"),  # identical release
        (_resolution("1.1.0", "c", package_reference="pkg:skill/other"), "different package"),
    ],
)
def test_r018_upgrade_rejects_non_upgrades_without_touching_the_installed_release(
    resolution: SkillPackageResolution, fragment: str
) -> None:
    lifecycle = _installed()

    result = _upgrade(lifecycle, resolution, operation_id="op-bad")

    assert result.outcome is SkillLifecycleOutcome.REJECTED
    assert fragment in result.reason_detail
    assert result.event_type is None
    binding = lifecycle.binding_metadata(_SKILL)
    assert binding is not None and binding.content_digest == _digest("a")
    assert lifecycle.registry.get(_SKILL).manifest.version == "1.0.0"


def test_r018_upgrade_never_goes_back_to_an_older_release_of_the_same_major() -> None:
    lifecycle = _installed()
    assert _upgrade(lifecycle, _resolution("1.2.0", "d")).outcome is SkillLifecycleOutcome.APPLIED

    result = _upgrade(lifecycle, _resolution("1.1.0", "b"), operation_id="op-downgrade")

    assert result.outcome is SkillLifecycleOutcome.REJECTED
    assert "not newer" in result.reason_detail
    binding = lifecycle.binding_metadata(_SKILL)
    assert binding is not None and binding.version_label == "1.2.0"


@pytest.mark.parametrize("failure", ["materialization", "binding_identity"])
def test_r018_failed_upgrade_restores_the_previous_release(failure: str) -> None:
    lifecycle = _installed()
    materializer = _Materializer(
        lifecycle.registry,
        fail=failure == "materialization",
        wrong_binding=failure == "binding_identity",
    )

    result = _upgrade(lifecycle, _resolution("1.1.0", "b"), materializer=materializer)

    assert result.outcome is SkillLifecycleOutcome.REJECTED
    assert "previous release restored" in result.reason_detail
    assert lifecycle.is_bound(_SKILL)
    binding = lifecycle.binding_metadata(_SKILL)
    assert binding is not None and binding.content_digest == _digest("a")
    assert lifecycle.registry.get(_SKILL).manifest.version == "1.0.0"
    assert is_skill_enabled(lifecycle.skill_profile, _SKILL)
    # a clean retry with the same operation id is still possible because nothing was recorded
    retry = _upgrade(lifecycle, _resolution("1.1.0", "b"))
    assert retry.outcome is SkillLifecycleOutcome.APPLIED


def test_r018_upgrade_requires_the_skill_and_the_host_profile() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id=_HOST)
    missing = _upgrade(lifecycle, _resolution("1.1.0", "b"))
    assert missing.outcome is SkillLifecycleOutcome.REJECTED
    assert "not installed" in missing.reason_detail

    installed = _installed()
    wrong_host = _upgrade(installed, _resolution("1.1.0", "b"), host_profile_id="other-host")
    assert wrong_host.outcome is SkillLifecycleOutcome.REJECTED
    assert "host_profile_id" in wrong_host.reason_detail


def test_r018_upgrade_works_through_the_catalog_materializer_pair_used_in_production_wiring() -> None:
    """ME-15 reference catalog: v1 -> v2 is a major change and must not upgrade in place."""
    lifecycle = SkillHostLifecycleService(host_profile_id=_HOST)
    provider = Me15SkillCatalogProvider()
    entry = next(item for item in provider.list_entries() if item.catalog_entry_id == ME15_CATALOG_ENTRY_ID)
    materializer = Me15SkillHostBindingMaterializer(lifecycle.registry, catalog_source_id=provider.catalog_source_id)
    ack = lifecycle.bind(
        operation_id="op-me15-install",
        host_profile_id=_HOST,
        resolved=provider.resolve_package(entry, version_selector="1.0.0"),
        materializer=materializer,
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.ACCEPTED

    result = lifecycle.upgrade_skill(
        operation_id="op-me15-upgrade",
        host_profile_id=_HOST,
        resolved=provider.resolve_package(entry, version_selector=ME15_VERSION_V2),
        materializer=materializer,
    )

    assert result.outcome is SkillLifecycleOutcome.REJECTED
    binding = lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
    assert binding is not None and binding.version_label == "1.0.0"


def test_r018_skill_lifecycle_is_separate_from_agent_lifecycle() -> None:
    import ast
    from pathlib import Path

    source = Path(SkillHostLifecycleService.__module__.replace(".", "/") + ".py")
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / source).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    assert not {name for name in imported if name.startswith("intergrax.agent_distribution")}
    assert not {name for name in imported if name.startswith("intergrax.runtime.registry")}
