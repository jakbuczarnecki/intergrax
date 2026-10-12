# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0 typed inventory model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class TenantDomain(StrEnum):
    IDENTITY_AUTH = "identity_auth"
    RUNTIME_REQUEST_EXECUTION = "runtime_request_execution"
    GOVERNANCE = "governance"
    TOOL_INVOCATION = "tool_invocation"
    PROVIDER_INVOCATION = "provider_invocation"
    INTEGRATION_RESOLUTION = "integration_resolution"
    CONFIGURATION_PROFILES = "configuration_profiles"
    DURABLE_STATE = "durable_state"
    RUNTIME_STATE = "runtime_state"
    CHECKPOINT = "checkpoint"
    RETRY = "retry"
    RESUME = "resume"
    RECOVERY = "recovery"
    REPLAY = "replay"
    BACKUP_RESTORE = "backup_restore"
    CACHE = "cache"
    MEMORY = "memory"
    RAG = "rag"
    VECTOR_INDEX = "vector_index"
    GRAPH_INDEX = "graph_index"
    EVENTS = "events"
    EVIDENCE = "evidence"
    TRACE = "trace"
    OBSERVABILITY = "observability"
    DIAGNOSTICS = "diagnostics"
    QUEUES = "queues"
    BACKGROUND_WORK = "background_work"
    CHILD_EXECUTION = "child_execution"
    EXTERNAL_OPERATIONS = "external_operations"
    CREDENTIALS_SECRETS = "credentials_secrets"
    PLUGINS = "plugins"
    PROVIDER_EXTENSIONS = "provider_extensions"
    COMPATIBILITY_ASSESSMENT = "compatibility_assessment"
    CONTROL_PLANE = "control_plane"
    CONTEXT_PROMPT = "context_prompt"
    PLATFORM_GLOBAL = "platform_global"


class TenantPropagationRole(StrEnum):
    CANONICAL_IDENTITY_SOURCE = "CANONICAL_IDENTITY_SOURCE"
    IDENTITY_CARRIER = "IDENTITY_CARRIER"
    IDENTITY_DERIVATION = "IDENTITY_DERIVATION"
    TENANT_VALIDATOR = "TENANT_VALIDATOR"
    TENANT_SCOPE_NARROWING = "TENANT_SCOPE_NARROWING"
    TENANT_SCOPED_LOOKUP = "TENANT_SCOPED_LOOKUP"
    TENANT_SCOPED_WRITE = "TENANT_SCOPED_WRITE"
    TENANT_SCOPED_CACHE = "TENANT_SCOPED_CACHE"
    TENANT_SCOPED_PROVIDER_SELECTION = "TENANT_SCOPED_PROVIDER_SELECTION"
    TENANT_SCOPED_CONFIGURATION = "TENANT_SCOPED_CONFIGURATION"
    TENANT_SCOPED_CREDENTIAL_REFERENCE = "TENANT_SCOPED_CREDENTIAL_REFERENCE"
    TENANT_SCOPED_EVIDENCE = "TENANT_SCOPED_EVIDENCE"
    TENANT_SCOPED_EVENT = "TENANT_SCOPED_EVENT"
    TENANT_SCOPED_RECOVERY = "TENANT_SCOPED_RECOVERY"
    TENANT_SCOPED_CHILD_OPERATION = "TENANT_SCOPED_CHILD_OPERATION"
    LEGAL_CROSS_TENANT_BOUNDARY = "LEGAL_CROSS_TENANT_BOUNDARY"
    GLOBAL_NON_TENANT_SURFACE = "GLOBAL_NON_TENANT_SURFACE"
    UNKNOWN = "UNKNOWN"


class TenantIdentitySource(StrEnum):
    CANONICAL_CONTRACT = "CANONICAL_CONTRACT"
    DERIVED_FROM_PARENT = "DERIVED_FROM_PARENT"
    DERIVED_FROM_REQUEST = "DERIVED_FROM_REQUEST"
    EXPLICIT_PARAMETER = "EXPLICIT_PARAMETER"
    OPTIONAL_ABSENT = "OPTIONAL_ABSENT"
    AMBIENT_DEFAULT_LITERAL = "AMBIENT_DEFAULT_LITERAL"
    GLOBAL_NON_TENANT = "GLOBAL_NON_TENANT"
    VALIDATED_ENVELOPE = "VALIDATED_ENVELOPE"
    UNKNOWN = "UNKNOWN"


class TenantPersistenceRole(StrEnum):
    TENANT_IN_KEY_OR_PREDICATE = "TENANT_IN_KEY_OR_PREDICATE"
    TENANT_VALIDATED_BEFORE_LOOKUP = "TENANT_VALIDATED_BEFORE_LOOKUP"
    GLOBAL_SHARED_BY_DESIGN = "GLOBAL_SHARED_BY_DESIGN"
    MISSING_ISOLATION_EVIDENCE = "MISSING_ISOLATION_EVIDENCE"
    EXPLICIT_CROSS_TENANT_CONTRACT = "EXPLICIT_CROSS_TENANT_CONTRACT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TenantAuthorityRole(StrEnum):
    TENANT_SCOPE_AUTHORITY = "TENANT_SCOPE_AUTHORITY"
    NON_TENANT_ADVISORY = "NON_TENANT_ADVISORY"
    CROSS_TENANT_GOVERNED = "CROSS_TENANT_GOVERNED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TenantFailClosedDisposition(StrEnum):
    FAIL_CLOSED_ON_MISSING = "FAIL_CLOSED_ON_MISSING"
    FAIL_CLOSED_ON_MISMATCH = "FAIL_CLOSED_ON_MISMATCH"
    SILENT_DEFAULT_SUBSTITUTION = "SILENT_DEFAULT_SUBSTITUTION"
    OPTIONAL_BYPASS = "OPTIONAL_BYPASS"
    NON_TENANT_SURFACE = "NON_TENANT_SURFACE"
    UNKNOWN = "UNKNOWN"


class TenantBlockerClass(StrEnum):
    IN_SCOPE_BLOCKER = "IN_SCOPE_BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED_FREEZE_DEBT"
    ENVIRONMENT_TEST_ISSUE = "ENVIRONMENT_TEST_ISSUE"
    ARCHITECTURE_DECISION_REQUIRED = "ARCHITECTURE_DECISION_REQUIRED"
    NONE = "NONE"


class PropagationRelation(StrEnum):
    SAME = "SAME"
    NARROWER = "NARROWER"
    EXPLICIT_CROSS_TENANT = "EXPLICIT_CROSS_TENANT"


class AdversarialProbeVerdict(StrEnum):
    EXPECT_FAIL_CLASSIFICATION = "EXPECT_FAIL_CLASSIFICATION"
    EXPECT_PASS_CLASSIFICATION = "EXPECT_PASS_CLASSIFICATION"


class AmbientDefaultSemantics(StrEnum):
    TENANT_REQUIRED_SILENT_DEFAULT = "TENANT_REQUIRED_SILENT_DEFAULT"
    GLOBAL_NON_TENANT_DEFAULT = "GLOBAL_NON_TENANT_DEFAULT"
    DEMO_OR_APPLICATION_LOCAL_DEFAULT = "DEMO_OR_APPLICATION_LOCAL_DEFAULT"
    COMPATIBILITY_DEFAULT = "COMPATIBILITY_DEFAULT"
    CONFIGURATION_DEFAULT_REQUIRING_VALIDATION = "CONFIGURATION_DEFAULT_REQUIRING_VALIDATION"
    UNKNOWN_DEFAULT_SEMANTICS = "UNKNOWN_DEFAULT_SEMANTICS"


class OptionalTenantSemantics(StrEnum):
    SAFE_INGRESS_REQUIRES_LATER = "SAFE_INGRESS_REQUIRES_LATER"
    LEGITIMATE_GLOBAL_SURFACE = "LEGITIMATE_GLOBAL_SURFACE"
    LEGACY_COMPATIBILITY_SEAM = "LEGACY_COMPATIBILITY_SEAM"
    UNSAFE_OPTIONAL_BYPASS = "UNSAFE_OPTIONAL_BYPASS"
    UNKNOWN = "UNKNOWN"


class PropagationEdgeStatus(StrEnum):
    PROVEN = "PROVEN"
    MISSING = "MISSING"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class TenantDiscoveryCandidate:
    surface_id: str
    module_path: str
    symbol: str
    discovery_kind: str
    line_no: int


@dataclass(frozen=True, slots=True)
class TenantSurfaceRecord:
    surface_id: str
    module_path: str
    symbol: str
    domain: TenantDomain
    tenant_required: bool
    identity_source: TenantIdentitySource
    semantic_owner: str
    propagation_role: TenantPropagationRole
    persistence_role: TenantPersistenceRole
    authority_role: TenantAuthorityRole
    fail_closed_behavior: TenantFailClosedDisposition
    cross_tenant_capable: bool
    cross_tenant_contract: str | None
    evidence_refs: tuple[str, ...]
    blocker_class: TenantBlockerClass = TenantBlockerClass.NONE
    blocker_note: str = ""


@dataclass(frozen=True, slots=True)
class TenantPropagationEdge:
    edge_id: str
    source_surface_id: str
    target_surface_id: str
    expected_relation: PropagationRelation
    frz_ten_criteria: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceItem:
    stage_id: str
    accepted_sha: str
    domain: str
    frz_ten_criteria: tuple[str, ...]
    historical_status: str
    current_head_replay_required: bool
    replay_note: str


@dataclass(frozen=True, slots=True)
class TenantOwnerAssignment:
    responsibility: str
    owner: str
    surfaces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TenantP0Matrix:
    production_files_scanned: int
    tenant_related_candidates: int
    tenant_required_surfaces: int
    global_non_tenant_surfaces: int
    canonical_identity_sources: int
    tenant_carriers: int
    derived_identity_carriers: int
    ambient_default_identity_sources: int
    unknown_identity_sources: int
    propagation_edges: int
    state_persistence_surfaces: int
    provider_config_surfaces: int
    credential_secret_surfaces: int
    evidence_trace_surfaces: int
    recovery_surfaces: int
    extension_surfaces: int
    legal_cross_tenant_boundaries: int
    implicit_default_candidates: int
    unknown_propagation_roles: int
    in_scope_blockers: int
    platform_escape_count: int


@dataclass(frozen=True, slots=True)
class ClosedWorldDiscoveryReport:
    module_paths_scanned: tuple[str, ...]
    raw_candidates: tuple[TenantDiscoveryCandidate, ...]
    unclassified_candidate_ids: frozenset[str]
