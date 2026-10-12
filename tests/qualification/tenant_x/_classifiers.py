# © Artur Czarnecki. All rights reserved.

"""TENANT-X surface classifiers — fail-closed; no UNKNOWN at inventory gate."""

from __future__ import annotations

from tests.qualification.tenant_x._types import (
    TenantAuthorityRole,
    TenantBlockerClass,
    TenantDiscoveryCandidate,
    TenantDomain,
    TenantFailClosedDisposition,
    TenantIdentitySource,
    TenantPersistenceRole,
    TenantPropagationRole,
    TenantSurfaceRecord,
)

_DOMAIN_PREFIX: tuple[tuple[str, TenantDomain], ...] = (
    ("intergrax/runtime/execution", TenantDomain.RUNTIME_REQUEST_EXECUTION),
    ("intergrax/runtime/checkpoint", TenantDomain.CHECKPOINT),
    ("intergrax/runtime/recovery", TenantDomain.RECOVERY),
    ("intergrax/runtime/replay", TenantDomain.REPLAY),
    ("intergrax/runtime/events", TenantDomain.EVENTS),
    ("intergrax/runtime/schema", TenantDomain.DURABLE_STATE),
    ("intergrax/governance", TenantDomain.GOVERNANCE),
    ("intergrax/tools", TenantDomain.TOOL_INVOCATION),
    ("intergrax/integrations", TenantDomain.INTEGRATION_RESOLUTION),
    ("intergrax/integrations/credentials", TenantDomain.CREDENTIALS_SECRETS),
    ("intergrax/memory", TenantDomain.MEMORY),
    ("intergrax/context", TenantDomain.CONTEXT_PROMPT),
    ("intergrax/contracts", TenantDomain.IDENTITY_AUTH),
    ("intergrax/core", TenantDomain.CONTROL_PLANE),
    ("intergrax/distributed", TenantDomain.CACHE),
    ("intergrax/collaborative_work", TenantDomain.EVENTS),
    ("intergrax/capability_metering", TenantDomain.OBSERVABILITY),
    ("agents/", TenantDomain.BACKGROUND_WORK),
    ("applications/", TenantDomain.EXTERNAL_OPERATIONS),
)

_GLOBAL_NON_TENANT_MARKERS: frozenset[str] = frozenset(
    {
        "health",
        "liveness",
        "readiness",
        "metrics_registry",
        "platform_version",
    }
)


def _domain_for_path(module_path: str) -> TenantDomain:
    for prefix, domain in _DOMAIN_PREFIX:
        if module_path.startswith(prefix):
            return domain
    if "vector" in module_path or "rag" in module_path:
        return TenantDomain.RAG
    if "graph" in module_path:
        return TenantDomain.GRAPH_INDEX
    if "queue" in module_path or "worker" in module_path:
        return TenantDomain.QUEUES
    if "plugin" in module_path:
        return TenantDomain.PLUGINS
    if "checkpoint" in module_path:
        return TenantDomain.CHECKPOINT
    if "retry" in module_path or "idempotency" in module_path:
        return TenantDomain.RETRY
    if "trace" in module_path or "evidence" in module_path:
        return TenantDomain.EVIDENCE
    if "observability" in module_path or "diagnostic" in module_path:
        return TenantDomain.OBSERVABILITY
    if "config" in module_path or "profile" in module_path:
        return TenantDomain.CONFIGURATION_PROFILES
    if "child" in module_path or "delegat" in module_path:
        return TenantDomain.CHILD_EXECUTION
    return TenantDomain.RUNTIME_REQUEST_EXECUTION


def _semantic_owner(module_path: str, domain: TenantDomain) -> str:
    if domain == TenantDomain.GOVERNANCE:
        return "GOVERNANCE"
    if domain == TenantDomain.RUNTIME_REQUEST_EXECUTION:
        return "EXECUTION"
    if domain in {TenantDomain.INTEGRATION_RESOLUTION, TenantDomain.PROVIDER_INVOCATION}:
        return "INTEGRATIONS"
    if domain == TenantDomain.CONFIGURATION_PROFILES:
        return "CONFIGURATION_REALIZATION"
    if domain in {TenantDomain.DURABLE_STATE, TenantDomain.CHECKPOINT, TenantDomain.RECOVERY}:
        return "STATE_RECOVERY"
    if domain in {TenantDomain.EVIDENCE, TenantDomain.TRACE}:
        return "TRACEABILITY"
    if domain == TenantDomain.MEMORY:
        return "MEMORY"
    if domain == TenantDomain.CONTEXT_PROMPT:
        return "CONTEXT_ENGINEERING"
    if domain == TenantDomain.CREDENTIALS_SECRETS:
        return "INTEGRATIONS_CREDENTIALS"
    if module_path.startswith("applications/"):
        return "APPLICATION_BOUNDARY"
    if module_path.startswith("agents/"):
        return "AGENTS_RUNTIME"
    return "PLATFORM_CORE"


def classify_candidate(candidate: TenantDiscoveryCandidate) -> TenantSurfaceRecord:
    domain = _domain_for_path(candidate.module_path)
    owner = _semantic_owner(candidate.module_path, domain)
    kind = candidate.discovery_kind

    if kind.startswith("canonical."):
        return TenantSurfaceRecord(
            surface_id=candidate.surface_id,
            module_path=candidate.module_path,
            symbol=candidate.symbol,
            domain=domain,
            tenant_required=True,
            identity_source=TenantIdentitySource.CANONICAL_CONTRACT,
            semantic_owner=owner,
            propagation_role=TenantPropagationRole.CANONICAL_IDENTITY_SOURCE,
            persistence_role=TenantPersistenceRole.NOT_APPLICABLE,
            authority_role=TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
            fail_closed_behavior=TenantFailClosedDisposition.FAIL_CLOSED_ON_MISSING,
            cross_tenant_capable=False,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:canonical_anchor",),
        )

    if kind.startswith("ambient_default"):
        return TenantSurfaceRecord(
            surface_id=candidate.surface_id,
            module_path=candidate.module_path,
            symbol=candidate.symbol,
            domain=domain,
            tenant_required=True,
            identity_source=TenantIdentitySource.AMBIENT_DEFAULT_LITERAL,
            semantic_owner=owner,
            propagation_role=TenantPropagationRole.IDENTITY_DERIVATION,
            persistence_role=TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
            authority_role=TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
            fail_closed_behavior=TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION,
            cross_tenant_capable=True,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:ambient_default_scan",),
            blocker_class=TenantBlockerClass.IN_SCOPE_BLOCKER,
            blocker_note="Ambient default tenant literal in production path",
        )

    if kind == "optional.tenant":
        role = TenantPropagationRole.IDENTITY_CARRIER
        identity = TenantIdentitySource.OPTIONAL_ABSENT
        fail_closed = TenantFailClosedDisposition.OPTIONAL_BYPASS
        blocker = TenantBlockerClass.TRACKED_FREEZE_DEBT
        note = "Optional tenant field — requires TENANT-X child continuity proof"
        if candidate.module_path.endswith("agent_runtime_io.py"):
            blocker = TenantBlockerClass.IN_SCOPE_BLOCKER
            note = "RuntimeRequest optional tenant_id until envelope conversion"
        return TenantSurfaceRecord(
            surface_id=candidate.surface_id,
            module_path=candidate.module_path,
            symbol=candidate.symbol,
            domain=domain,
            tenant_required=True,
            identity_source=identity,
            semantic_owner=owner,
            propagation_role=role,
            persistence_role=TenantPersistenceRole.TENANT_VALIDATED_BEFORE_LOOKUP,
            authority_role=TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
            fail_closed_behavior=fail_closed,
            cross_tenant_capable=False,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:optional_tenant",),
            blocker_class=blocker,
            blocker_note=note,
        )

    if kind == "validator.marker":
        return TenantSurfaceRecord(
            surface_id=candidate.surface_id,
            module_path=candidate.module_path,
            symbol=candidate.symbol,
            domain=domain,
            tenant_required=True,
            identity_source=TenantIdentitySource.VALIDATED_ENVELOPE,
            semantic_owner=owner,
            propagation_role=TenantPropagationRole.TENANT_VALIDATOR,
            persistence_role=TenantPersistenceRole.TENANT_VALIDATED_BEFORE_LOOKUP,
            authority_role=TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
            fail_closed_behavior=TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
            cross_tenant_capable=False,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:validator_marker",),
        )

    if any(marker in candidate.symbol.lower() for marker in _GLOBAL_NON_TENANT_MARKERS):
        return TenantSurfaceRecord(
            surface_id=candidate.surface_id,
            module_path=candidate.module_path,
            symbol=candidate.symbol,
            domain=TenantDomain.PLATFORM_GLOBAL,
            tenant_required=False,
            identity_source=TenantIdentitySource.GLOBAL_NON_TENANT,
            semantic_owner="OBSERVABILITY",
            propagation_role=TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE,
            persistence_role=TenantPersistenceRole.GLOBAL_SHARED_BY_DESIGN,
            authority_role=TenantAuthorityRole.NON_TENANT_ADVISORY,
            fail_closed_behavior=TenantFailClosedDisposition.NON_TENANT_SURFACE,
            cross_tenant_capable=False,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:global_non_tenant",),
        )

    propagation = TenantPropagationRole.IDENTITY_CARRIER
    persistence = TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE
    identity = TenantIdentitySource.EXPLICIT_PARAMETER
    tenant_required = True

    if "broker" in candidate.module_path or "credential" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_CREDENTIAL_REFERENCE
        domain = TenantDomain.CREDENTIALS_SECRETS
        persistence = TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE
    elif "memory" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_CACHE
        domain = TenantDomain.MEMORY
    elif "vector" in candidate.module_path or "rag" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_LOOKUP
        domain = TenantDomain.VECTOR_INDEX
    elif "config" in candidate.module_path or "profile" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_CONFIGURATION
        domain = TenantDomain.CONFIGURATION_PROFILES
    elif "provider" in candidate.module_path or "integrations" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION
        domain = TenantDomain.PROVIDER_INVOCATION
    elif "event" in candidate.module_path or "evidence" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_EVIDENCE
        domain = TenantDomain.EVIDENCE
    elif "checkpoint" in candidate.module_path or "recovery" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_RECOVERY
        domain = TenantDomain.RECOVERY
    elif "child" in candidate.module_path:
        propagation = TenantPropagationRole.TENANT_SCOPED_CHILD_OPERATION
        domain = TenantDomain.CHILD_EXECUTION
    elif kind == "function.param.tenant":
        identity = TenantIdentitySource.DERIVED_FROM_REQUEST

    if candidate.module_path.startswith("applications/") and kind == "class.field.tenant":
        blocker = TenantBlockerClass.TRACKED_FREEZE_DEBT
        blocker_note = "Application demo schema default tenant — not platform canonical identity"
    else:
        blocker = TenantBlockerClass.NONE
        blocker_note = ""

    return TenantSurfaceRecord(
        surface_id=candidate.surface_id,
        module_path=candidate.module_path,
        symbol=candidate.symbol,
        domain=domain,
        tenant_required=tenant_required,
        identity_source=identity,
        semantic_owner=owner,
        propagation_role=propagation,
        persistence_role=persistence,
        authority_role=TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
        fail_closed_behavior=TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
        cross_tenant_capable=False,
        cross_tenant_contract=None,
        evidence_refs=("TENANT-X-P0:mechanical_discovery",),
        blocker_class=blocker,
        blocker_note=blocker_note,
    )
