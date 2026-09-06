# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.core.version_binding import (
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver

pytestmark = pytest.mark.unit


def _manifest(
    skill_id: str,
    version: str = "1.0.0",
    *,
    requires_skills: tuple[str, ...] = (),
) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        description=skill_id,
        version=version,
        requires_skills=requires_skills,
    )


def test_root_pin_passes_when_registry_matches() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    resolver = SkillResolver(registry)
    pack = resolver.resolve_skills([_manifest("a.pack", "1.0.0")])
    assert pack.resolved_skills[0].version == "1.0.0"
    assert pack.resolved_skills[0].resolution_mode is SkillVersionResolutionMode.PINNED


def test_root_pin_fails_on_registry_version_mismatch() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "2.0.0"))
    resolver = SkillResolver(registry)
    with pytest.raises(SkillResolutionError, match="version mismatch"):
        resolver.resolve_skills([_manifest("a.pack", "1.0.0")])


def test_root_pin_fails_on_unknown_skill() -> None:
    resolver = SkillResolver(SkillRegistry())
    with pytest.raises(SkillResolutionError, match="Unknown skill_id"):
        resolver.resolve_skills([_manifest("missing.pack", "1.0.0")])


def test_validate_skills_checks_version() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    resolver = SkillResolver(registry)
    resolver.validate_skills([_manifest("a.pack", "1.0.0")])
    with pytest.raises(SkillResolutionError, match="version mismatch"):
        resolver.validate_skills([_manifest("a.pack", "2.0.0")])


def test_transitive_dependency_records_materialized_version() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("b.pack", "2.3.0"))
    registry.register(
        _manifest("a.pack", "1.0.0", requires_skills=("b.pack",)),
    )
    pack = SkillResolver(registry).resolve_skills([_manifest("a.pack", "1.0.0")])
    assert tuple(ref.qualified_id for ref in pack.resolved_skills) == (
        "b.pack@2.3.0",
        "a.pack@1.0.0",
    )
    assert pack.resolved_skills[0].role is ResolvedSkillRole.TRANSITIVE
    assert pack.resolved_skills[0].resolution_mode is SkillVersionResolutionMode.MATERIALIZED


def test_root_and_transitive_conflict_fails_closed() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("b.pack", "2.0.0"))
    registry.register(
        _manifest("a.pack", "1.0.0", requires_skills=("b.pack",)),
    )
    resolver = SkillResolver(registry)
    with pytest.raises(SkillResolutionError, match="version mismatch"):
        resolver.resolve_skills(
            [
                _manifest("a.pack", "1.0.0"),
                _manifest("b.pack", "1.0.0"),
            ],
        )


def test_conflicting_root_versions_fail_closed() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    resolver = SkillResolver(registry)
    with pytest.raises(SkillResolutionError, match="conflicting root version"):
        resolver.resolve_skills(
            [
                _manifest("a.pack", "1.0.0"),
                _manifest("a.pack", "2.0.0"),
            ],
        )


def test_duplicate_same_root_version_is_deduplicated() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    pack = SkillResolver(registry).resolve_skills(
        [
            _manifest("a.pack", "1.0.0"),
            _manifest("a.pack", "1.0.0"),
        ],
    )
    assert pack.skill_ids == ("a.pack",)


def test_registry_replacement_breaks_future_pinned_resolution() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    resolver = SkillResolver(registry)
    first = resolver.resolve_skills([_manifest("a.pack", "1.0.0")])
    registry.register_or_replace(_manifest("a.pack", "2.0.0"))
    with pytest.raises(SkillResolutionError, match="version mismatch"):
        resolver.resolve_skills([_manifest("a.pack", "1.0.0")])
    assert first.resolved_skills[0].version == "1.0.0"


def test_snapshot_digest_is_deterministic_and_immutable() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    resolver = SkillResolver(registry)
    first = resolver.resolve_skills([_manifest("a.pack", "1.0.0")])
    second = resolver.resolve_skills([_manifest("a.pack", "1.0.0")])
    assert first.snapshot_digest == second.snapshot_digest
    assert first.snapshot_digest.startswith("sha256:")


def test_resolve_without_manifest_uses_materialized_versions() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "2.0.0"))
    pack = SkillResolver(registry).resolve(["a.pack"])
    assert pack.resolved_skills[0].version == "2.0.0"
    assert pack.resolved_skills[0].resolution_mode is SkillVersionResolutionMode.MATERIALIZED
