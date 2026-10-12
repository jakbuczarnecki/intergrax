# © Artur Czarnecki. All rights reserved.

"""TENANT-X surface classifiers — bounded positive evidence; fail-closed UNKNOWN default."""

from __future__ import annotations

from tests.qualification.tenant_x._discovery import discover_candidates_from_source
from tests.qualification.tenant_x._evidence import evidence_kinds, has_kind
from tests.qualification.tenant_x._types import (
    AmbientDefaultSemantics,
    OptionalTenantSemantics,
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

_EVID_SQL_PREDICATE = "TENANT_IN_SQL_PREDICATE"
_EVID_SQL_MISSING = "TENANT_MISSING_IN_SQL_QUERY"
_EVID_SUBSCRIPT_KEY = "TENANT_IN_SUBSCRIPT_KEY"
_EVID_SUBSCRIPT_MISSING = "TENANT_MISSING_IN_SUBSCRIPT"
_EVID_DEFAULT_PROTECTED = "DEFAULT_FLOWS_TO_PROTECTED_OP"
_EVID_APP_LOCAL = "APPLICATION_LOCAL_MODULE"
_EVID_COMPAT = "COMPATIBILITY_MODULE"
_EVID_GLOBAL_SYMBOL = "GLOBAL_NON_TENANT_SYMBOL"
_EVID_VALIDATOR_INVOKED = "VALIDATOR_INVOKED_BEFORE_LOOKUP"
_EVID_LOOKUP_NO_VALIDATOR = "PROTECTED_LOOKUP_WITHOUT_VALIDATOR"

# Longer / more specific prefixes must appear before broader parents.
_DOMAIN_PREFIX: tuple[tuple[str, TenantDomain], ...] = tuple(
    sorted(
        (
            ("intergrax/integrations/credentials", TenantDomain.CREDENTIALS_SECRETS),
            ("intergrax/runtime/execution", TenantDomain.RUNTIME_REQUEST_EXECUTION),
            ("intergrax/runtime/checkpoint", TenantDomain.CHECKPOINT),
            ("intergrax/runtime/recovery", TenantDomain.RECOVERY),
            ("intergrax/runtime/replay", TenantDomain.REPLAY),
            ("intergrax/runtime/events", TenantDomain.EVENTS),
            ("intergrax/runtime/schema", TenantDomain.DURABLE_STATE),
            ("intergrax/governance", TenantDomain.GOVERNANCE),
            ("intergrax/tools", TenantDomain.TOOL_INVOCATION),
            ("intergrax/integrations", TenantDomain.INTEGRATION_RESOLUTION),
            ("intergrax/memory", TenantDomain.MEMORY),
            ("intergrax/context", TenantDomain.CONTEXT_PROMPT),
            ("intergrax/contracts", TenantDomain.IDENTITY_AUTH),
            ("intergrax/core", TenantDomain.CONTROL_PLANE),
            ("intergrax/distributed", TenantDomain.CACHE),
            ("intergrax/collaborative_work", TenantDomain.EVENTS),
            ("intergrax/capability_metering", TenantDomain.OBSERVABILITY),
            ("agents/", TenantDomain.BACKGROUND_WORK),
            ("applications/", TenantDomain.EXTERNAL_OPERATIONS),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
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

_CANONICAL_ANCHOR_PROFILES: dict[str, tuple[TenantPropagationRole, TenantAuthorityRole, TenantIdentitySource, bool, TenantFailClosedDisposition, str]] = {
    "canonical.runtime_request": (
        TenantPropagationRole.IDENTITY_CARRIER,
        TenantAuthorityRole.NON_TENANT_ADVISORY,
        TenantIdentitySource.OPTIONAL_ABSENT,
        False,
        TenantFailClosedDisposition.UNKNOWN,
        "ingress carrier; tenant required only after admission/conversion evidence",
    ),
    "canonical.execution_admission": (
        TenantPropagationRole.TENANT_VALIDATOR,
        TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
        TenantIdentitySource.VALIDATED_ENVELOPE,
        True,
        TenantFailClosedDisposition.FAIL_CLOSED_ON_MISSING,
        "root execution admission validates tenant envelope",
    ),
    "canonical.governance_event": (
        TenantPropagationRole.IDENTITY_CARRIER,
        TenantAuthorityRole.NON_TENANT_ADVISORY,
        TenantIdentitySource.DERIVED_FROM_REQUEST,
        True,
        TenantFailClosedDisposition.UNKNOWN,
        "governance audit carries tenant; not global identity authority",
    ),
    "canonical.functional_evidence_tenant": (
        TenantPropagationRole.TENANT_VALIDATOR,
        TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
        TenantIdentitySource.VALIDATED_ENVELOPE,
        True,
        TenantFailClosedDisposition.FAIL_CLOSED_ON_MISSING,
        "functional evidence requires tenant from execution context",
    ),
    "canonical.runtime_inspection_query": (
        TenantPropagationRole.TENANT_VALIDATOR,
        TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
        TenantIdentitySource.VALIDATED_ENVELOPE,
        True,
        TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
        "runtime inspection query validates tenant scope",
    ),
    "canonical.credential_broker": (
        TenantPropagationRole.TENANT_SCOPED_CREDENTIAL_REFERENCE,
        TenantAuthorityRole.NON_TENANT_ADVISORY,
        TenantIdentitySource.EXPLICIT_PARAMETER,
        True,
        TenantFailClosedDisposition.UNKNOWN,
        "credential broker reference; provider consumption edge not proven here",
    ),
    "canonical.memory_scope_authority": (
        TenantPropagationRole.TENANT_VALIDATOR,
        TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
        TenantIdentitySource.VALIDATED_ENVELOPE,
        True,
        TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
        "memory scope authorization validates tenant before access",
    ),
    "canonical.tool_runtime_scope": (
        TenantPropagationRole.TENANT_SCOPE_NARROWING,
        TenantAuthorityRole.NON_TENANT_ADVISORY,
        TenantIdentitySource.DERIVED_FROM_PARENT,
        True,
        TenantFailClosedDisposition.UNKNOWN,
        "tool runtime scope narrows parent tenant; isolation proof requires tool path evidence",
    ),
    "canonical.child_execution_runner": (
        TenantPropagationRole.TENANT_SCOPED_CHILD_OPERATION,
        TenantAuthorityRole.NON_TENANT_ADVISORY,
        TenantIdentitySource.DERIVED_FROM_PARENT,
        True,
        TenantFailClosedDisposition.UNKNOWN,
        "child execution propagates parent tenant; checkpoint/recovery edge not proven here",
    ),
}


def domain_for_path(module_path: str) -> TenantDomain:
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
    if "trace" in module_path:
        return TenantDomain.TRACE
    if "evidence" in module_path:
        return TenantDomain.EVIDENCE
    if "observability" in module_path or "diagnostic" in module_path:
        return TenantDomain.DIAGNOSTICS
    if "config" in module_path or "profile" in module_path:
        return TenantDomain.CONFIGURATION_PROFILES
    if "child" in module_path or "delegat" in module_path:
        return TenantDomain.CHILD_EXECUTION
    if "recovery" in module_path or "resume" in module_path:
        return TenantDomain.RECOVERY
    if "provider" in module_path:
        return TenantDomain.PROVIDER_EXTENSIONS
    return TenantDomain.RUNTIME_REQUEST_EXECUTION


def _semantic_owner(module_path: str, domain: TenantDomain) -> str:
    if domain == TenantDomain.GOVERNANCE:
        return "GOVERNANCE"
    if domain == TenantDomain.RUNTIME_REQUEST_EXECUTION:
        return "EXECUTION"
    if domain in {TenantDomain.INTEGRATION_RESOLUTION, TenantDomain.PROVIDER_INVOCATION, TenantDomain.PROVIDER_EXTENSIONS}:
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


def _unknown_surface(
    candidate: TenantDiscoveryCandidate,
    domain: TenantDomain,
    owner: str,
    *,
    evidence: str = "TENANT-X-P0:unknown_semantics",
    note: str = "No bounded positive semantic rule matched",
) -> TenantSurfaceRecord:
    return TenantSurfaceRecord(
        surface_id=candidate.surface_id,
        module_path=candidate.module_path,
        symbol=candidate.symbol,
        domain=domain,
        tenant_required=False,
        identity_source=TenantIdentitySource.UNKNOWN,
        semantic_owner=owner,
        propagation_role=TenantPropagationRole.UNKNOWN,
        persistence_role=TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
        authority_role=TenantAuthorityRole.NOT_APPLICABLE,
        fail_closed_behavior=TenantFailClosedDisposition.UNKNOWN,
        cross_tenant_capable=False,
        cross_tenant_contract=None,
        evidence_refs=(evidence,),
        blocker_class=TenantBlockerClass.TRACKED_FREEZE_DEBT,
        blocker_note=note,
    )


def _persistence_from_evidence(
    candidate: TenantDiscoveryCandidate,
) -> tuple[TenantPersistenceRole, tuple[str, ...], TenantBlockerClass, str]:
    kinds = evidence_kinds(candidate.evidence)
    extra_refs: list[str] = []
    blocker = TenantBlockerClass.NONE
    note = ""

    if _EVID_VALIDATOR_INVOKED in kinds:
        return (
            TenantPersistenceRole.TENANT_VALIDATED_BEFORE_LOOKUP,
            ("TENANT-X-P0:ast:validated_before_lookup",),
            TenantBlockerClass.NONE,
            "Validator invocation precedes protected lookup in same function",
        )
    if _EVID_SQL_PREDICATE in kinds or _EVID_SUBSCRIPT_KEY in kinds:
        return (
            TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE,
            ("TENANT-X-P0:ast:tenant_in_key_or_predicate", *extra_refs),
            blocker,
            note,
        )
    if _EVID_LOOKUP_NO_VALIDATOR in kinds:
        return (
            TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
            ("TENANT-X-P0:ast:lookup_without_validator",),
            TenantBlockerClass.NONE,
            "Validator exists in module but protected lookup not proven guarded",
        )
    if _EVID_SQL_MISSING in kinds or _EVID_SUBSCRIPT_MISSING in kinds:
        blocker = TenantBlockerClass.IN_SCOPE_BLOCKER
        note = "Tenant represented but storage/query lacks tenant isolation evidence"
        return (
            TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
            ("TENANT-X-P0:ast:missing_isolation",),
            blocker,
            note,
        )
    return (
        TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
        ("TENANT-X-P0:identity_represented_only",),
        blocker,
        note,
    )


def _classify_ambient_default(
    candidate: TenantDiscoveryCandidate,
    domain: TenantDomain,
    owner: str,
    kind: str,
) -> TenantSurfaceRecord:
    semantics = AmbientDefaultSemantics.UNKNOWN_DEFAULT_SEMANTICS
    ev = candidate.evidence
    if has_kind(ev, _EVID_DEFAULT_PROTECTED):
        semantics = AmbientDefaultSemantics.TENANT_REQUIRED_SILENT_DEFAULT
    elif has_kind(ev, _EVID_APP_LOCAL) or candidate.module_path.startswith("applications/"):
        semantics = AmbientDefaultSemantics.DEMO_OR_APPLICATION_LOCAL_DEFAULT
    elif has_kind(ev, _EVID_GLOBAL_SYMBOL) or any(m in candidate.symbol.lower() for m in _GLOBAL_NON_TENANT_MARKERS):
        semantics = AmbientDefaultSemantics.GLOBAL_NON_TENANT_DEFAULT
    elif has_kind(ev, _EVID_COMPAT) or "compat" in candidate.module_path or "legacy" in candidate.module_path:
        semantics = AmbientDefaultSemantics.COMPATIBILITY_DEFAULT
    elif "config" in candidate.module_path or "profile" in candidate.module_path:
        semantics = AmbientDefaultSemantics.CONFIGURATION_DEFAULT_REQUIRING_VALIDATION

    blocker = TenantBlockerClass.NONE
    note = f"ambient_default:{semantics.value}"
    tenant_required = False
    identity = TenantIdentitySource.AMBIENT_DEFAULT_LITERAL
    fail_closed = TenantFailClosedDisposition.UNKNOWN
    cross_tenant = False

    if semantics == AmbientDefaultSemantics.TENANT_REQUIRED_SILENT_DEFAULT:
        tenant_required = True
        fail_closed = TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION
        cross_tenant = True
        blocker = TenantBlockerClass.IN_SCOPE_BLOCKER
        note = "Ambient default flows into protected operation (source evidence)"
    elif semantics == AmbientDefaultSemantics.UNKNOWN_DEFAULT_SEMANTICS:
        blocker = TenantBlockerClass.TRACKED_FREEZE_DEBT
        note = "IN-SCOPE CLASSIFICATION BLOCKER: ambient default semantics unknown"

    return TenantSurfaceRecord(
        surface_id=candidate.surface_id,
        module_path=candidate.module_path,
        symbol=candidate.symbol,
        domain=domain,
        tenant_required=tenant_required,
        identity_source=identity,
        semantic_owner=owner,
        propagation_role=TenantPropagationRole.IDENTITY_DERIVATION,
        persistence_role=TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
        authority_role=TenantAuthorityRole.NOT_APPLICABLE,
        fail_closed_behavior=fail_closed,
        cross_tenant_capable=cross_tenant,
        cross_tenant_contract=None,
        evidence_refs=(f"TENANT-X-P0:ambient:{semantics.value}",),
        blocker_class=blocker,
        blocker_note=note,
    )


def _classify_optional_tenant(
    candidate: TenantDiscoveryCandidate,
    domain: TenantDomain,
    owner: str,
) -> TenantSurfaceRecord:
    semantics = OptionalTenantSemantics.UNKNOWN
    blocker = TenantBlockerClass.TRACKED_FREEZE_DEBT
    note = "Optional tenant field — semantics require admission/conversion evidence"

    if candidate.module_path.endswith("agent_runtime_io.py"):
        semantics = OptionalTenantSemantics.SAFE_INGRESS_REQUIRES_LATER
        blocker = TenantBlockerClass.NONE
        note = "RuntimeRequest optional tenant_id evaluated at envelope conversion — not automatic debt"
    elif candidate.module_path.startswith("applications/"):
        semantics = OptionalTenantSemantics.LEGACY_COMPATIBILITY_SEAM
        note = "Application-local optional tenant seam"
    elif any(m in candidate.module_path for m in ("health", "metrics", "observability")):
        semantics = OptionalTenantSemantics.LEGITIMATE_GLOBAL_SURFACE
        blocker = TenantBlockerClass.NONE
        note = "Optional tenant on non-tenant observability surface"

    tenant_required = semantics in {
        OptionalTenantSemantics.UNSAFE_OPTIONAL_BYPASS,
        OptionalTenantSemantics.UNKNOWN,
    }
    fail_closed = TenantFailClosedDisposition.UNKNOWN
    if semantics == OptionalTenantSemantics.UNSAFE_OPTIONAL_BYPASS:
        fail_closed = TenantFailClosedDisposition.OPTIONAL_BYPASS
        blocker = TenantBlockerClass.IN_SCOPE_BLOCKER

    return TenantSurfaceRecord(
        surface_id=candidate.surface_id,
        module_path=candidate.module_path,
        symbol=candidate.symbol,
        domain=domain,
        tenant_required=tenant_required,
        identity_source=TenantIdentitySource.OPTIONAL_ABSENT,
        semantic_owner=owner,
        propagation_role=TenantPropagationRole.IDENTITY_CARRIER,
        persistence_role=TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE,
        authority_role=TenantAuthorityRole.NOT_APPLICABLE,
        fail_closed_behavior=fail_closed,
        cross_tenant_capable=False,
        cross_tenant_contract=None,
        evidence_refs=(f"TENANT-X-P0:optional:{semantics.value}",),
        blocker_class=blocker,
        blocker_note=note,
    )


def _classify_class_field_tenant(
    candidate: TenantDiscoveryCandidate,
    domain: TenantDomain,
    owner: str,
) -> TenantSurfaceRecord:
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

    if "credential" in candidate.module_path and not candidate.discovery_kind.startswith("canonical"):
        return _unknown_surface(
            candidate,
            TenantDomain.CREDENTIALS_SECRETS,
            "INTEGRATIONS_CREDENTIALS",
            note="credentials path without credential tenant semantics",
        )

    persistence, evidence, blocker, persistence_note = _persistence_from_evidence(candidate)
    note = persistence_note or "Class field proves identity representation only"
    propagation = TenantPropagationRole.IDENTITY_CARRIER
    if persistence == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE:
        propagation = TenantPropagationRole.TENANT_SCOPED_LOOKUP
    elif persistence == TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE and evidence == (
        "TENANT-X-P0:identity_represented_only",
    ):
        propagation = TenantPropagationRole.UNKNOWN

    return TenantSurfaceRecord(
        surface_id=candidate.surface_id,
        module_path=candidate.module_path,
        symbol=candidate.symbol,
        domain=domain,
        tenant_required=False,
        identity_source=TenantIdentitySource.UNKNOWN,
        semantic_owner=owner,
        propagation_role=propagation,
        persistence_role=persistence,
        authority_role=TenantAuthorityRole.NOT_APPLICABLE,
        fail_closed_behavior=TenantFailClosedDisposition.UNKNOWN,
        cross_tenant_capable=False,
        cross_tenant_contract=None,
        evidence_refs=evidence,
        blocker_class=blocker,
        blocker_note=note,
    )


def _classify_function_param_tenant(
    candidate: TenantDiscoveryCandidate,
    domain: TenantDomain,
    owner: str,
) -> TenantSurfaceRecord:
    persistence, ev_refs, blocker, note = _persistence_from_evidence(candidate)
    if persistence == TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE and ev_refs == (
        "TENANT-X-P0:identity_represented_only",
    ):
        persistence = TenantPersistenceRole.NOT_APPLICABLE
        ev_refs = ("TENANT-X-P0:explicit_parameter_carrier_only",)
        note = "Function parameter is explicit tenant carrier only"
    propagation = TenantPropagationRole.IDENTITY_CARRIER
    if persistence == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE:
        propagation = TenantPropagationRole.TENANT_SCOPED_LOOKUP
    return TenantSurfaceRecord(
        surface_id=candidate.surface_id,
        module_path=candidate.module_path,
        symbol=candidate.symbol,
        domain=domain,
        tenant_required=False,
        identity_source=TenantIdentitySource.EXPLICIT_PARAMETER,
        semantic_owner=owner,
        propagation_role=propagation,
        persistence_role=persistence,
        authority_role=TenantAuthorityRole.NOT_APPLICABLE,
        fail_closed_behavior=TenantFailClosedDisposition.UNKNOWN,
        cross_tenant_capable=False,
        cross_tenant_contract=None,
        evidence_refs=ev_refs,
        blocker_class=blocker,
        blocker_note=note,
    )


def classify_candidate(candidate: TenantDiscoveryCandidate) -> TenantSurfaceRecord:
    domain = domain_for_path(candidate.module_path)
    owner = _semantic_owner(candidate.module_path, domain)
    kind = candidate.discovery_kind

    if kind.startswith("canonical."):
        profile = _CANONICAL_ANCHOR_PROFILES.get(kind)
        if profile is None:
            return _unknown_surface(candidate, domain, owner, note=f"unknown canonical kind {kind}")
        propagation, authority, identity, tenant_required, fail_closed, note = profile
        return TenantSurfaceRecord(
            surface_id=candidate.surface_id,
            module_path=candidate.module_path,
            symbol=candidate.symbol,
            domain=domain,
            tenant_required=tenant_required,
            identity_source=identity,
            semantic_owner=owner,
            propagation_role=propagation,
            persistence_role=TenantPersistenceRole.NOT_APPLICABLE,
            authority_role=authority,
            fail_closed_behavior=fail_closed,
            cross_tenant_capable=False,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:canonical_anchor_profile", note),
            blocker_class=TenantBlockerClass.NONE,
            blocker_note=note,
        )

    if kind.startswith("ambient_default"):
        return _classify_ambient_default(candidate, domain, owner, kind)

    if kind == "optional.tenant":
        return _classify_optional_tenant(candidate, domain, owner)

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
            persistence_role=TenantPersistenceRole.NOT_APPLICABLE,
            authority_role=TenantAuthorityRole.TENANT_SCOPE_AUTHORITY,
            fail_closed_behavior=TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
            cross_tenant_capable=False,
            cross_tenant_contract=None,
            evidence_refs=("TENANT-X-P0:validator_marker_only",),
            blocker_class=TenantBlockerClass.NONE,
            blocker_note="Validator marker proves validator surface only — not lookup guard",
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

    if kind == "class.field.tenant":
        return _classify_class_field_tenant(candidate, domain, owner)

    if kind == "function.param.tenant":
        return _classify_function_param_tenant(candidate, domain, owner)

    return _unknown_surface(candidate, domain, owner)


def classify_source_fixture(module_path: str, source: str) -> tuple[TenantSurfaceRecord, ...]:
    """Classify fixture source via production discovery + classifier (no probe shortcuts)."""
    candidates = discover_candidates_from_source(module_path, source)
    return tuple(classify_candidate(c) for c in candidates)


def uses_generic_safe_fallback(record: TenantSurfaceRecord) -> bool:
    """True when record asserts safe tenant semantics without bounded evidence."""
    if record.evidence_refs and record.evidence_refs[0] == "TENANT-X-P0:mechanical_discovery":
        return True
    joined_refs = "".join(record.evidence_refs)
    if (
        record.persistence_role == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE
        and "ast:tenant_in_key_or_predicate" not in joined_refs
        and "tenant_predicate_positive" not in joined_refs
        and record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
    ):
        return True
    if (
        record.fail_closed_behavior == TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH
        and record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
        and "explicit_parameter_carrier_only" in "".join(record.evidence_refs)
    ):
        return True
    if record.propagation_role in {
        TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION,
        TenantPropagationRole.TENANT_SCOPED_RECOVERY,
        TenantPropagationRole.TENANT_SCOPED_CREDENTIAL_REFERENCE,
    } and "canonical_anchor_profile" not in "".join(record.evidence_refs):
        if "path" in record.blocker_note.lower():
            return True
    return False


def is_path_name_only_classification(record: TenantSurfaceRecord) -> bool:
    path_markers = ("provider", "credential", "checkpoint", "recovery", "config")
    if not any(m in record.module_path for m in path_markers):
        return False
    joined = "".join(record.evidence_refs)
    return (
        "bounded" not in joined
        and "validator" not in joined
        and "canonical_anchor" not in joined
        and record.propagation_role
        in {
            TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION,
            TenantPropagationRole.TENANT_SCOPED_RECOVERY,
            TenantPropagationRole.TENANT_SCOPED_CREDENTIAL_REFERENCE,
        }
    )
