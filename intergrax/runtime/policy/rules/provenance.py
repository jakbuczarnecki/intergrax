# © Artur Czarnecki. All rights reserved.

"""Policy bundle provenance DTOs (BLOCK B / CAND-008)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from intergrax.core.plugins.discovery import EntryPointSpec
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule

PolicyRulesSourceKind = Literal["inline", "file", "mixed"]


@dataclass(frozen=True, slots=True)
class PolicyHandlerProvenance:
    """Entry-point metadata for one accepted policy rule handler."""

    rule_id: str
    ep_name: str
    ep_value: str
    distribution: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyBundleProvenance:
    """
    Deterministic provenance for a declarative policy bundle.

    Does not claim signing, attestation, or production qualification.
    """

    source_kind: PolicyRulesSourceKind
    rules_path: str | None
    rules_digest_sha256: str
    handler_provenance: tuple[PolicyHandlerProvenance, ...]
    rejected_handler_ids: tuple[str, ...] = ()


def digest_inline_rules(rules: tuple[DeclarativePolicyRule, ...]) -> str:
    """Canonical SHA-256 digest for inline validated rules."""
    payload = [rule.model_dump(mode="json") for rule in rules]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def digest_policy_rules_file(path: Path) -> str:
    """SHA-256 digest of exact policy rules file bytes."""
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def handler_provenance_from_spec(rule_id: str, spec: EntryPointSpec) -> PolicyHandlerProvenance:
    return PolicyHandlerProvenance(
        rule_id=rule_id,
        ep_name=spec.name,
        ep_value=spec.value,
        distribution=spec.distribution,
    )
