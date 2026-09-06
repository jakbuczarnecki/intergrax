# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.core.version_binding import (
    ResolvedSkillRef,
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.snapshot_digest import compute_resolved_skill_pack_digest

pytestmark = pytest.mark.unit


def _ref(
    skill_id: str,
    version: str,
    *,
    resolution_mode: SkillVersionResolutionMode = SkillVersionResolutionMode.PINNED,
    role: ResolvedSkillRole = ResolvedSkillRole.ROOT,
) -> ResolvedSkillRef:
    return ResolvedSkillRef(
        skill_id=skill_id,
        version=version,
        qualified_id=f"{skill_id}@{version}",
        resolution_mode=resolution_mode,
        role=role,
    )


def _legacy_delimiter_canonical(resolved_skills: tuple[ResolvedSkillRef, ...]) -> str:
    parts: list[str] = []
    for ref in resolved_skills:
        parts.append(
            "|".join(
                (
                    ref.skill_id,
                    ref.version,
                    ref.resolution_mode.value,
                    ref.role.value,
                )
            )
        )
    return "resolved_skill_pack.v1:" + ";".join(parts)


def test_legacy_field_boundary_collision_pre_hash() -> None:
    ref_a = _ref("a|b", "c")
    ref_b = _ref("a", "b|c")
    assert _legacy_delimiter_canonical((ref_a,)) == _legacy_delimiter_canonical((ref_b,))
    assert compute_resolved_skill_pack_digest((ref_a,)) != compute_resolved_skill_pack_digest(
        (ref_b,)
    )


def test_legacy_record_boundary_collision_pre_hash() -> None:
    single = (
        _ref(
            "a|b|pinned|pinned|root;c",
            "d",
            resolution_mode=SkillVersionResolutionMode.MATERIALIZED,
            role=ResolvedSkillRole.TRANSITIVE,
        ),
    )
    split = (
        _ref("a|b", "pinned"),
        _ref(
            "c",
            "d",
            resolution_mode=SkillVersionResolutionMode.MATERIALIZED,
            role=ResolvedSkillRole.TRANSITIVE,
        ),
    )
    assert _legacy_delimiter_canonical(single) == _legacy_delimiter_canonical(split)
    assert compute_resolved_skill_pack_digest(single) != compute_resolved_skill_pack_digest(split)


def test_unicode_fields_are_deterministic_and_distinct() -> None:
    ref_ascii = _ref("skill.alpha", "1.0.0")
    ref_unicode = _ref("skill.\u03b1", "1.0.0")
    ref_other = _ref("skill.\u03b2", "1.0.0")
    first = compute_resolved_skill_pack_digest((ref_unicode,))
    second = compute_resolved_skill_pack_digest((ref_unicode,))
    assert first == second
    assert first.startswith("sha256:")
    assert compute_resolved_skill_pack_digest((ref_ascii,)) != first
    assert compute_resolved_skill_pack_digest((ref_other,)) != first


def test_order_changes_digest() -> None:
    ref_a = _ref("a.pack", "1.0.0")
    ref_b = _ref("b.pack", "2.0.0", role=ResolvedSkillRole.TRANSITIVE)
    assert compute_resolved_skill_pack_digest((ref_a, ref_b)) != compute_resolved_skill_pack_digest(
        (ref_b, ref_a)
    )


def test_same_input_produces_same_digest() -> None:
    ref = _ref("a.pack", "1.0.0")
    assert compute_resolved_skill_pack_digest((ref,)) == compute_resolved_skill_pack_digest((ref,))


def test_empty_snapshot_has_stable_digest() -> None:
    first = compute_resolved_skill_pack_digest(())
    second = compute_resolved_skill_pack_digest(())
    assert first == second
    assert first.startswith("sha256:")


def test_role_changes_digest() -> None:
    ref_root = _ref("a.pack", "1.0.0", role=ResolvedSkillRole.ROOT)
    ref_transitive = _ref("a.pack", "1.0.0", role=ResolvedSkillRole.TRANSITIVE)
    assert compute_resolved_skill_pack_digest((ref_root,)) != compute_resolved_skill_pack_digest(
        (ref_transitive,)
    )


def test_resolution_mode_changes_digest() -> None:
    ref_pinned = _ref("a.pack", "1.0.0", resolution_mode=SkillVersionResolutionMode.PINNED)
    ref_materialized = _ref(
        "a.pack",
        "1.0.0",
        resolution_mode=SkillVersionResolutionMode.MATERIALIZED,
    )
    assert compute_resolved_skill_pack_digest((ref_pinned,)) != compute_resolved_skill_pack_digest(
        (ref_materialized,)
    )


def test_digest_uses_sha256_hex_format() -> None:
    digest = compute_resolved_skill_pack_digest(())
    assert digest.startswith("sha256:")
    hex_part = digest.removeprefix("sha256:")
    assert len(hex_part) == 64
    int(hex_part, 16)
