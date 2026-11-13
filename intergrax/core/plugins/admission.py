# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared immutable DTOs for domain plugin bootstrap evidence (ENTERPRISE-2)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from intergrax.core.distribution.package_identity import normalize_distribution_package_name
from intergrax.core.plugins.discovery import EntryPointLoadResult, EntryPointSpec

if TYPE_CHECKING:
    from intergrax.agent_distribution.catalog import CatalogSourceIdentity
    from intergrax.agent_distribution.package_attestation import AgentPackageAttestationVerifier
    from intergrax.agent_distribution.trust import (
        AgentDeliverySource,
        AgentPackageTrustPolicy,
        AgentPackageTrustRevocationState,
    )


class PluginAdmissionReasonCode(StrEnum):
    """Operator-visible admission rejection classification."""

    INVALID_TARGET_TYPE = "invalid_target_type"
    PLUGIN_ID_COLLISION = "plugin_id_collision"
    SHIPPED_ID_COLLISION = "shipped_id_collision"
    ALREADY_REGISTERED = "already_registered"
    PLUGIN_ID_SKIPPED = "plugin_id_skipped"
    NOT_IN_ALLOWLIST = "not_in_allowlist"
    PLUGIN_NOT_SELECTED = "plugin_not_selected"
    PRODUCTION_ADMISSION_DENIED = "production_admission_denied"
    INVALID_POLICY_CONTRIBUTION_SOURCE = "invalid_policy_contribution_source"
    POLICY_HANDLER_BINDING_MISSING = "policy_handler_binding_missing"
    POLICY_HANDLER_PROVENANCE_MISMATCH = "policy_handler_provenance_mismatch"
    POLICY_CONFIGURATION_CONTRACT_BINDING_MISSING = "policy_configuration_contract_binding_missing"
    POLICY_CONFIGURATION_CONTRACT_ID_MISMATCH = "policy_configuration_contract_id_mismatch"
    CONFIGURATION_CONTRACT_BUILTIN_RESERVED = "configuration_contract_builtin_reserved"
    UNRESOLVED_PACKAGE_IDENTITY = "unresolved_package_identity"
    MANIFEST_BINDING_UNAVAILABLE = "manifest_binding_unavailable"
    MANIFEST_INVALID = "manifest_invalid"
    MANIFEST_CAPABILITY_BINDING_MISSING = "manifest_capability_binding_missing"
    CAPABILITY_ID_MISMATCH = "capability_id_mismatch"
    PLUGIN_IDENTITY_MISMATCH = "plugin_identity_mismatch"
    METADATA_PLUGIN_ID_COLLISION = "metadata_plugin_id_collision"
    REQUESTED_PLUGIN_LOCATOR_NOT_FOUND = "requested_plugin_locator_not_found"
    REQUESTED_PLUGIN_LOCATOR_AMBIGUOUS = "requested_plugin_locator_ambiguous"
    MANIFEST_PLUGIN_ID_MISMATCH = "manifest_plugin_id_mismatch"
    PLUGIN_PROVENANCE_MISSING = "plugin_provenance_missing"
    PLUGIN_PROVENANCE_INVALID = "plugin_provenance_invalid"
    PLUGIN_PROVENANCE_DENIED = "plugin_provenance_denied"


@dataclass(frozen=True, slots=True)
class PluginAdmissionRejection:
    """Admission/validation rejection — distinct from entry-point load failure."""

    spec: EntryPointSpec
    reason_code: PluginAdmissionReasonCode
    reason: str
    plugin_id: str | None = None
    fail_closed: bool = True


@dataclass(frozen=True, slots=True)
class DomainPluginLoadReport:
    """Per-bootstrap domain evidence. Not a unified operator inventory (F006)."""

    group: str
    accepted: tuple[EntryPointSpec, ...]
    rejected: tuple[PluginAdmissionRejection, ...]
    failed: tuple[EntryPointLoadResult, ...]
    registered_count: int

    @property
    def critical_bootstrap_acceptable(self) -> bool:
        """True when no load failures and no fail-closed admission rejections."""
        if self.failed:
            return False
        return not any(item.fail_closed for item in self.rejected)

    def to_audit_dict(self) -> dict[str, object]:
        """JSON-friendly metadata without live plugin targets."""
        return {
            "group": self.group,
            "registered_count": self.registered_count,
            "critical_bootstrap_acceptable": self.critical_bootstrap_acceptable,
            "accepted": tuple(_spec_audit(spec) for spec in self.accepted),
            "rejected": tuple(_rejection_audit(item) for item in self.rejected),
            "failed": tuple(_failed_audit(item) for item in self.failed),
        }

    @classmethod
    def empty(cls, group: str) -> DomainPluginLoadReport:
        return cls(
            group=group,
            accepted=(),
            rejected=(),
            failed=(),
            registered_count=0,
        )


def _entry_point_spec_identity(spec: EntryPointSpec) -> tuple[str, str]:
    return (spec.name, spec.value)


def merge_domain_plugin_load_reports(
    *reports: DomainPluginLoadReport,
) -> DomainPluginLoadReport:
    """Merge bootstrap evidence from multiple discovery passes (same plugin group only)."""
    if not reports:
        raise ValueError("merge_domain_plugin_load_reports requires at least one report")
    group = reports[0].group
    for report in reports[1:]:
        if report.group != group:
            raise ValueError(f"Cannot merge plugin load reports with differing groups: {group!r} vs {report.group!r}")

    accepted_by_key: dict[tuple[str, str], EntryPointSpec] = {}
    rejected_by_key: dict[tuple[str, str], PluginAdmissionRejection] = {}
    failed_by_key: dict[tuple[str, str], EntryPointLoadResult] = {}

    for report in reports:
        for spec in report.accepted:
            accepted_by_key[_entry_point_spec_identity(spec)] = spec
        for item in report.rejected:
            rejected_by_key[_entry_point_spec_identity(item.spec)] = item
        for item in report.failed:
            failed_by_key[_entry_point_spec_identity(item.spec)] = item

    accepted = tuple(sorted(accepted_by_key.values(), key=lambda spec: (spec.name, spec.value)))
    rejected = tuple(
        sorted(
            rejected_by_key.values(),
            key=lambda item: (item.spec.name, item.spec.value),
        )
    )
    failed = tuple(
        sorted(
            failed_by_key.values(),
            key=lambda item: (item.spec.name, item.spec.value),
        )
    )
    return DomainPluginLoadReport(
        group=group,
        accepted=accepted,
        rejected=rejected,
        failed=failed,
        registered_count=len(accepted),
    )


def _spec_audit(spec: EntryPointSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "group": spec.group,
        "value": spec.value,
        "distribution": spec.distribution,
    }


def _rejection_audit(item: PluginAdmissionRejection) -> dict[str, object]:
    payload = _spec_audit(item.spec)
    payload["reason_code"] = item.reason_code.value
    payload["reason"] = item.reason
    payload["plugin_id"] = item.plugin_id
    payload["fail_closed"] = item.fail_closed
    return payload


def _failed_audit(item: EntryPointLoadResult) -> dict[str, object]:
    payload = _spec_audit(item.spec)
    error = item.error
    payload["error_type"] = type(error).__name__ if error is not None else None
    payload["error"] = str(error) if error is not None else None
    return payload


# ---------------------------------------------------------------------------
# PDC-D6-R016 / OD-PDC-SCOPE-07 = A - entry-point provenance admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PluginProvenanceAttestation:
    """Publisher attestation for one installed plugin distribution (secret-free).

    Same Ed25519 statement as agent packages: it binds ``distribution``, ``package_version``
    and ``package_digest`` to ``publisher_id`` / ``key_id``. Verification key material is
    supplied by the key provider of the verifier, never by the attestation itself.
    """

    distribution: str
    package_version: str
    package_digest: str
    publisher_id: str
    attestation_id: str
    key_id: str
    signature_b64: str


def _default_installed_version(distribution: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalized_or_none(value: str) -> str | None:
    try:
        return normalize_distribution_package_name(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class PluginProvenanceAdmission:
    """Entry-point provenance admission built on the agent package trust mechanism.

    Reuses ``AgentPackageTrustPolicy``, ``AgentPackageTrustCoordinator`` and the Ed25519
    attestation verifier from ``intergrax.agent_distribution``; it adds no second trust or
    signature authority. Strictness comes from the policy itself:
    ``forbid_unsigned_or_unqualified=True`` (the default) rejects any entry point without a
    verifiable attestation; ``False`` is the explicit lab policy that admits unsigned plugins
    while still rejecting plugins whose attestation is present but invalid.

    The attested digest is bound to the publisher by the signature; this admission does not
    re-hash the installed files, it checks the installed distribution version instead.
    """

    trust_policy: AgentPackageTrustPolicy
    attestation_verifier: AgentPackageAttestationVerifier
    attestations: Mapping[str, PluginProvenanceAttestation] = field(default_factory=dict)
    catalog_source: CatalogSourceIdentity | None = None
    delivery_source: AgentDeliverySource | None = None
    revocation_state: AgentPackageTrustRevocationState | None = None
    installed_version_resolver: Callable[[str], str | None] = _default_installed_version
    clock: Callable[[], datetime] = _utc_now

    @staticmethod
    def _reject(
        spec: EntryPointSpec,
        code: PluginAdmissionReasonCode,
        reason: str,
    ) -> PluginAdmissionRejection:
        return PluginAdmissionRejection(spec=spec, reason_code=code, reason=reason, fail_closed=True)

    def _attestation_for(self, spec: EntryPointSpec) -> PluginProvenanceAttestation | None:
        if not spec.distribution:
            return None
        wanted = _normalized_or_none(spec.distribution)
        if wanted is None:
            return None
        for key, item in self.attestations.items():
            if _normalized_or_none(key) == wanted:
                return item
        return None

    def evaluate(self, spec: EntryPointSpec) -> PluginAdmissionRejection | None:
        """Return a rejection, or ``None`` when ``spec`` may be loaded and registered."""
        from intergrax.agent_distribution.catalog import CatalogProviderKind, CatalogSourceIdentity
        from intergrax.agent_distribution.errors import AgentPackageAttestationError
        from intergrax.agent_distribution.identity import AgentPackageIdentity
        from intergrax.agent_distribution.package_attestation import (
            AgentPackageAttestationAlgorithm,
            AgentPackageAttestationVerificationRequest,
        )
        from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
        from intergrax.agent_distribution.trust import (
            AgentDeliverySource,
            AgentPackageQualificationResult,
            AgentPublisherIdentity,
        )

        policy = self.trust_policy
        attestation = self._attestation_for(spec)
        if attestation is None:
            if policy.forbid_unsigned_or_unqualified:
                return self._reject(
                    spec,
                    PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING,
                    f"entry point has no verifiable publisher attestation (distribution={spec.distribution!r})",
                )
            return None

        installed = self.installed_version_resolver(attestation.distribution)
        if installed is None or installed != attestation.package_version:
            return self._reject(
                spec,
                PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID,
                "installed distribution version does not match the attested version",
            )
        try:
            identity = AgentPackageIdentity(
                distribution_package_id=attestation.distribution,
                package_version=attestation.package_version,
                package_digest=attestation.package_digest,
            )
            evidence = self.attestation_verifier.verify_qualification_evidence(
                AgentPackageAttestationVerificationRequest(
                    package_identity=identity,
                    publisher_id=attestation.publisher_id,
                    attestation_id=attestation.attestation_id,
                    key_id=attestation.key_id,
                    algorithm=AgentPackageAttestationAlgorithm.ED25519,
                    signature_b64=attestation.signature_b64,
                )
            )
        except (AgentPackageAttestationError, ValueError) as exc:
            code = getattr(exc, "reason_code", None) or "attestation_invalid"
            return self._reject(
                spec,
                PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID,
                f"publisher attestation failed verification ({code})",
            )

        delivery = self.delivery_source or AgentDeliverySource.ORG_REGISTRY
        source = self.catalog_source or CatalogSourceIdentity(
            catalog_source_id="intergrax.plugin_entry_points",
            provider_kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
        )
        publisher = AgentPublisherIdentity(publisher_id=attestation.publisher_id)
        now = self.clock()
        qualification = AgentPackageQualificationResult(
            publisher=publisher,
            status=policy.effective_required_qualification_status,
            evidence=(cast("Any", evidence),),
            reason="plugin entry-point publisher attestation verified",
            delivery_source=delivery,
            qualified_at=now,
        )
        decision = AgentPackageTrustCoordinator(
            attestation_verifier=self.attestation_verifier,
        ).evaluate(
            package_identity=identity,
            catalog_source=source,
            delivery_source=delivery,
            publisher=publisher,
            policy=policy,
            qualification=qualification,
            evidence_package_digest=identity.package_digest,
            evidence_id=attestation.attestation_id,
            revocation_state=self.revocation_state,
            evaluated_at=now,
        )
        if not decision.installable:
            return self._reject(
                spec,
                PluginAdmissionReasonCode.PLUGIN_PROVENANCE_DENIED,
                f"trust policy denied the plugin ({decision.reason_code.value})",
            )
        return None

    def partition(
        self,
        specs: tuple[EntryPointSpec, ...],
    ) -> tuple[tuple[EntryPointSpec, ...], tuple[PluginAdmissionRejection, ...]]:
        """Split ``specs`` into admitted specs and rejections (before any target is imported)."""
        admitted: list[EntryPointSpec] = []
        rejected: list[PluginAdmissionRejection] = []
        for spec in specs:
            rejection = self.evaluate(spec)
            if rejection is None:
                admitted.append(spec)
            else:
                rejected.append(rejection)
        return tuple(admitted), tuple(rejected)
