# © Artur Czarnecki. All rights reserved.

"""Composition-time validation for runtime invariant rule packs."""

from __future__ import annotations

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantCompositionError,
    RuntimeInvariantRule,
    RuntimeInvariantRulePack,
    RuntimeInvariantRunner,
    runtime_invariant_rule_sort_key,
)
from intergrax.runtime.invariants.default_runner import DefaultRuntimeInvariantRunner


def validate_runtime_invariant_rule_packs(
    rule_packs: tuple[RuntimeInvariantRulePack, ...],
) -> tuple[RuntimeInvariantRule, ...]:
    """Fail fast on duplicate pack or rule identities; return sorted rules."""
    pack_ids: dict[str, str] = {}
    rule_ids: dict[str, tuple[str, str]] = {}
    rules: list[RuntimeInvariantRule] = []

    for pack in rule_packs:
        existing_pack_version = pack_ids.get(pack.pack_id)
        if existing_pack_version is not None:
            raise RuntimeInvariantCompositionError(
                f"duplicate runtime invariant pack_id {pack.pack_id!r}",
            )
        pack_ids[pack.pack_id] = pack.pack_version
        for rule in pack.rules:
            if rule.domain != pack.domain:
                raise RuntimeInvariantCompositionError(
                    f"rule {rule.rule_id!r} domain {rule.domain!s} "
                    f"does not match pack {pack.pack_id!r} domain {pack.domain!s}",
                )
            prior = rule_ids.get(rule.rule_id)
            if prior is not None:
                raise RuntimeInvariantCompositionError(
                    f"duplicate runtime invariant rule_id {rule.rule_id!r}",
                )
            rule_ids[rule.rule_id] = (rule.rule_version, pack.pack_id)
            rules.append(rule)

    return tuple(sorted(rules, key=runtime_invariant_rule_sort_key))


def compose_default_runtime_invariant_runner(
    rule_packs: tuple[RuntimeInvariantRulePack, ...],
) -> RuntimeInvariantRunner:
    """Default runner factory — composition layer owns concrete implementation."""
    rules = validate_runtime_invariant_rule_packs(rule_packs)
    return DefaultRuntimeInvariantRunner(rule_packs=rule_packs, rules=rules)


__all__ = [
    "compose_default_runtime_invariant_runner",
    "validate_runtime_invariant_rule_packs",
]
