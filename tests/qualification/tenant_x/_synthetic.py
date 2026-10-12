# © Artur Czarnecki. All rights reserved.

"""In-memory adversarial probes (TENANT-X-P0 §14–15) — source fixtures only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.tenant_x._authority_semantics import (
    build_authority_mechanisms,
    build_authority_review_metrics,
    classify_authority_duplicate_mechanisms,
    resolve_owner_disposition,
)
from tests.qualification.tenant_x._classifiers import classify_source_fixture
from tests.qualification.tenant_x._types import (
    AdversarialProbeVerdict,
    TenantAuthorityOwnerDisposition,
    TenantAuthorityRole,
    TenantAuthoritySemanticRole,
    TenantDomain,
    TenantBlockerClass,
    TenantCarrierFamily,
    TenantCarrierProvenanceRequirement,
    TenantFailClosedDisposition,
    TenantIdentitySource,
    TenantPersistenceRole,
    TenantPropagationRole,
    TenantSurfaceRecord,
)


class SyntheticProbeId(StrEnum):
    A_CROSS_TENANT_STATE_KEY = "A"
    B_CHILD_WIDENS_TENANT = "B"
    C_MISSING_TENANT_DEFAULT = "C"
    D_CROSS_TENANT_PROVIDER_CONFIG = "D"
    E_CROSS_TENANT_CREDENTIAL = "E"
    F_CROSS_TENANT_EVIDENCE = "F"
    G_CROSS_TENANT_CHECKPOINT_RECOVERY = "G"
    H_PLUGIN_REWRITES_TENANT = "H"
    I_LEGAL_CROSS_TENANT_CONTRACT = "I"
    J_GLOBAL_HEALTH_METRIC = "J"
    K_UNKNOWN_TENANT_FIELD = "K"
    L_FUNCTION_PARAM_ONLY = "L"
    M_SILENT_TENANT_DEFAULT = "M"
    N_HARMLESS_GLOBAL_METADATA = "N"
    O_CREDENTIALS_PATH_ONLY = "O"
    P_STORAGE_IGNORES_TENANT = "P"
    Q_ACTUAL_TENANT_PREDICATE = "Q"
    R_VALIDATOR_NOT_ON_LOOKUP = "R"
    S_VALIDATOR_BEFORE_LOOKUP = "S"
    T_UNRELATED_DICT_SUBSCRIPT = "T"
    U_GLOBAL_REFERENCE_SQL = "U"
    V_TENANT_STORE_MISSING_KEY = "V"
    W_TENANT_STORE_TENANT_KEY = "W"
    X_VALIDATOR_AFTER_LOOKUP = "X"
    Y_VALIDATOR_BEFORE_TENANT_KEY = "Y"
    Z1_PASSIVE_DTO = "Z1"
    Z2_EXECUTION_BOUNDARY = "Z2"
    Z3_STATE_DTO = "Z3"
    Z4_APPLICATION_SCHEMA = "Z4"
    Z5_EVIDENCE_EVENT = "Z5"
    Z6_FAMILY_UNKNOWN = "Z6"
    AA_VALIDATOR_ONLY = "AA"
    AB_PARENT_PROPAGATION = "AB"
    AC_ILLEGAL_REWRITE = "AC"
    AD_SCOPE_NARROWING = "AD"
    AE_SCOPE_WIDENING = "AE"
    AF_SILENT_DEFAULT = "AF"
    AG_CREDENTIAL_SAME = "AG"
    AH_CREDENTIAL_MISMATCH = "AH"
    AI_CHECKPOINT_SAME = "AI"
    AJ_CHECKPOINT_SUBSTITUTION = "AJ"
    AK_ONE_OWNER_TWO_CALL_SITES = "AK"
    AL_SAME_OWNER_DIFFERENT_DECISIONS = "AL"
    AM_COMPETING_OWNERS_SAME_DECISION = "AM"
    AN_PLATFORM_CORE_UNRESOLVED = "AN"
    AO_OWNER_CONFIRMED = "AO"
    AP_SAME_DECISION_DOMAIN_FALLBACK = "AP"
    AQ_EXPLICIT_DIFFERENT_SCOPES = "AQ"
    AR_EXPLICIT_SAME_SCOPE_COMPETING = "AR"
    AS_SAME_MECHANISM_CALL_SITES = "AS"


@dataclass(frozen=True, slots=True)
class SyntheticAdversarialScenario:
    probe_id: SyntheticProbeId
    description: str
    tenant_a: str
    tenant_b: str
    expected_verdict: AdversarialProbeVerdict


@dataclass(frozen=True, slots=True)
class ClassifierProbeScenario:
    probe_id: SyntheticProbeId
    module_path: str
    source: str
    surface_selector: str


@dataclass(frozen=True, slots=True)
class AuthorityMechanismProbeScenario:
    probe_id: SyntheticProbeId
    description: str


SYNTHETIC_ADVERSARIAL_SCENARIOS: tuple[SyntheticAdversarialScenario, ...] = (
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.A_CROSS_TENANT_STATE_KEY,
        description="tenant A request → tenant B state key",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.B_CHILD_WIDENS_TENANT,
        description="tenant A authority → child tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.C_MISSING_TENANT_DEFAULT,
        description='missing tenant → "default"',
        tenant_a="",
        tenant_b="default",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.D_CROSS_TENANT_PROVIDER_CONFIG,
        description="tenant A config → provider binding tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.E_CROSS_TENANT_CREDENTIAL,
        description="tenant A credential ref → tenant B provider",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.F_CROSS_TENANT_EVIDENCE,
        description="tenant A evidence → tenant B assessment",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.G_CROSS_TENANT_CHECKPOINT_RECOVERY,
        description="tenant A checkpoint → recovery tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.H_PLUGIN_REWRITES_TENANT,
        description="plugin receives tenant A → returns tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.I_LEGAL_CROSS_TENANT_CONTRACT,
        description="explicit legal cross-tenant typed+authorized contract",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_PASS_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.J_GLOBAL_HEALTH_METRIC,
        description="global non-tenant health metric",
        tenant_a="",
        tenant_b="",
        expected_verdict=AdversarialProbeVerdict.EXPECT_PASS_CLASSIFICATION,
    ),
)

_SOURCE_K = """
class ProbeK:
    tenant_id: str
"""

_SOURCE_L = """
def fn(tenant_id: str):
    return tenant_id
"""

_SOURCE_M = """
class Worker:
    tenant_id: str = "default"

    def fetch(self, item_id: str):
        return db.execute(
            "SELECT value FROM items WHERE tenant_id = ? AND item_id = ?",
            (self.tenant_id, item_id),
        )
"""

_SOURCE_N = """
class health_status:
    tenant_id: str
"""

_SOURCE_O = """
class Broker:
    tenant_id: str
"""

_SOURCE_P = """
class Store:
    tenant_id: str

    def load(self, item_id: str):
        return db.execute(
            "SELECT value FROM items WHERE item_id = ?",
            (item_id,),
        )
"""

_SOURCE_Q = """
class Repo:
    tenant_id: str

    def load(self, tenant_id: str, item_id: str):
        return db.execute(
            "SELECT value FROM items WHERE tenant_id = ? AND item_id = ?",
            (tenant_id, item_id),
        )
"""

_SOURCE_R = """
def require_tenant_id(value: str) -> str:
    if not value:
        raise ValueError("tenant_id is required")
    return value

def load(item_id: str, tenant_id: str):
    return db.execute(
        "SELECT value FROM items WHERE item_id = ?",
        (item_id,),
    )
"""

_SOURCE_S = """
def require_tenant_id(value: str) -> str:
    if not value:
        raise ValueError("tenant_id is required")
    return value

def load(item_id: str, tenant_id: str):
    tenant_id = require_tenant_id(tenant_id)
    return db.execute(
        "SELECT value FROM items WHERE tenant_id = ? AND item_id = ?",
        (tenant_id, item_id),
    )
"""

_SOURCE_T = """
class Service:
    tenant_id: str

    def run(self, name: str):
        return self.handlers[name]
"""

_SOURCE_U = """
def load_country(tenant_id: str, code: str):
    return db.execute(
        "SELECT name FROM countries WHERE code = ?",
        (code,),
    )
"""

_SOURCE_V = """
class Service:
    tenant_id: str
    store: dict

    def load(self, item_id: str):
        return self.store[item_id]
"""

_SOURCE_W = """
class Service:
    tenant_id: str
    store: dict

    def load(self, tenant_id: str, item_id: str):
        return self.store[(tenant_id, item_id)]
"""

_SOURCE_X = """
def require_tenant_id(value: str) -> str:
    if not value:
        raise ValueError("tenant_id is required")
    return value

tenant_store = {}

def load(tenant_id: str, item_id: str):
    value = tenant_store[item_id]
    require_tenant_id(tenant_id)
    return value
"""

_SOURCE_Y = """
def require_tenant_id(value: str) -> str:
    if not value:
        raise ValueError("tenant_id is required")
    return value

tenant_store = {}

def load(tenant_id: str, item_id: str):
    tenant_id = require_tenant_id(tenant_id)
    return tenant_store[(tenant_id, item_id)]
"""

CLASSIFIER_PROBE_SCENARIOS: tuple[ClassifierProbeScenario, ...] = (
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.K_UNKNOWN_TENANT_FIELD,
        module_path="synthetic/probes/k_unknown.py",
        source=_SOURCE_K,
        surface_selector="ProbeK",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.L_FUNCTION_PARAM_ONLY,
        module_path="synthetic/probes/l_param.py",
        source=_SOURCE_L,
        surface_selector="fn",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.M_SILENT_TENANT_DEFAULT,
        module_path="intergrax/runtime/execution/m_silent_default_fixture.py",
        source=_SOURCE_M,
        surface_selector="ambient_default_field",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.N_HARMLESS_GLOBAL_METADATA,
        module_path="synthetic/probes/n_global.py",
        source=_SOURCE_N,
        surface_selector="health_status",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.O_CREDENTIALS_PATH_ONLY,
        module_path="intergrax/integrations/credentials/o_path_fixture.py",
        source=_SOURCE_O,
        surface_selector="Broker",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.P_STORAGE_IGNORES_TENANT,
        module_path="synthetic/probes/p_storage.py",
        source=_SOURCE_P,
        surface_selector="Store",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Q_ACTUAL_TENANT_PREDICATE,
        module_path="synthetic/probes/q_predicate.py",
        source=_SOURCE_Q,
        surface_selector="Repo",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.R_VALIDATOR_NOT_ON_LOOKUP,
        module_path="synthetic/probes/r_validator_gap.py",
        source=_SOURCE_R,
        surface_selector="load",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.S_VALIDATOR_BEFORE_LOOKUP,
        module_path="synthetic/probes/s_validator_guard.py",
        source=_SOURCE_S,
        surface_selector="load",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.T_UNRELATED_DICT_SUBSCRIPT,
        module_path="synthetic/probes/t_unrelated_dict.py",
        source=_SOURCE_T,
        surface_selector="Service",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.U_GLOBAL_REFERENCE_SQL,
        module_path="synthetic/probes/u_global_sql.py",
        source=_SOURCE_U,
        surface_selector="load_country",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.V_TENANT_STORE_MISSING_KEY,
        module_path="synthetic/probes/v_tenant_store_missing.py",
        source=_SOURCE_V,
        surface_selector="Service",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.W_TENANT_STORE_TENANT_KEY,
        module_path="synthetic/probes/w_tenant_store_key.py",
        source=_SOURCE_W,
        surface_selector="load",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.X_VALIDATOR_AFTER_LOOKUP,
        module_path="synthetic/probes/x_validator_after.py",
        source=_SOURCE_X,
        surface_selector="load",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Y_VALIDATOR_BEFORE_TENANT_KEY,
        module_path="synthetic/probes/y_validator_before_key.py",
        source=_SOURCE_Y,
        surface_selector="load",
    ),
)

_SOURCE_Z1 = """
from dataclasses import dataclass

@dataclass
class EvidenceRecord:
    tenant_id: str
    value: str
"""

_SOURCE_Z2 = """
class RuntimeAdmissionEnvelope:
    tenant_id: str
    request_id: str
"""

_SOURCE_Z3 = """
class CheckpointRow:
    tenant_id: str
    checkpoint_id: str
"""

_SOURCE_Z4 = """
class AppTenantPayload:
    tenant_id: str
    payload: str
"""

_SOURCE_Z5 = """
class RuntimeEventRecord:
    tenant_id: str
    event_type: str
"""

_SOURCE_Z6 = """
class UnclassifiedCarrier:
    tenant_id: str
"""

_SOURCE_AA = """
def require_tenant_id(tenant_id: str) -> str:
    if not tenant_id:
        raise ValueError()
    return tenant_id
"""

_SOURCE_AB = """
class ChildRequest:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

def build_child(parent):
    return ChildRequest(tenant_id=parent.tenant_id)
"""

_SOURCE_AC = """
class ChildRequest:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

def build_bad():
    return ChildRequest(tenant_id="other-tenant")
"""

_SOURCE_AD = """
def narrow_scope(parent_scope, child_scope):
    if child_scope.tenant_id != parent_scope.tenant_id:
        raise ValueError()
    return child_scope
"""

_SOURCE_AE = """
def widen_scope(parent_scope, child_scope):
    child_scope.tenant_id = parent_scope.tenant_id
    child_scope.extra_tenant = "widened"
    return child_scope
"""

_SOURCE_AF = """
def resolve_tenant(request):
    tenant_id = request.tenant_id or "default"
    return tenant_id
"""

_SOURCE_AG = """
def use_credential(scope, grant):
    assert scope.tenant_id == grant.tenant_id
    return scope.tenant_id
"""

_SOURCE_AH = """
def bad_credential(scope_a, grant_a, scope_b):
    if scope_b.tenant_id != grant_a.tenant_id:
        raise ValueError()
    return scope_b.tenant_id
"""

_SOURCE_AI = """
class Checkpoint:
    tenant_id: str

def restore(checkpoint):
    return checkpoint.tenant_id
"""

_SOURCE_AJ = """
class Checkpoint:
    tenant_id: str

def resume(checkpoint, caller_tenant_id: str):
    return caller_tenant_id
"""

def _synthetic_mint_surface(
    *,
    surface_id: str,
    module_path: str,
    symbol: str,
    domain: TenantDomain,
    semantic_owner: str,
    evidence_refs: tuple[str, ...],
) -> TenantSurfaceRecord:
    return TenantSurfaceRecord(
        surface_id=surface_id,
        module_path=module_path,
        symbol=symbol,
        domain=domain,
        tenant_required=True,
        identity_source=TenantIdentitySource.EXPLICIT_PARAMETER,
        semantic_owner=semantic_owner,
        propagation_role=TenantPropagationRole.IDENTITY_DERIVATION,
        persistence_role=TenantPersistenceRole.NOT_APPLICABLE,
        authority_role=TenantAuthorityRole.NOT_APPLICABLE,
        fail_closed_behavior=TenantFailClosedDisposition.UNKNOWN,
        cross_tenant_capable=False,
        cross_tenant_contract=None,
        evidence_refs=evidence_refs,
        authority_semantic_role=TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY,
    )


def _authority_r3r1_probe_surfaces(probe_id: SyntheticProbeId) -> tuple[TenantSurfaceRecord, ...]:
    shared_scope = "TENANT-X-P0:authority_scope:synthetic.r3r1.shared"
    shared_mech = "TENANT-X-P0:authority_mechanism:shared_sanctioned_rewrite"
    root_decision = "TENANT-X-P0:authority:decision:ROOT_TENANT_SELECTION"
    recovery_decision = "TENANT-X-P0:authority:decision:RECOVERY_TENANT_RESTORE"
    mint_evidence = ("TENANT-X-P0:mint:literal_rewrite_kwarg",)

    if probe_id == SyntheticProbeId.AK_ONE_OWNER_TWO_CALL_SITES:
        module = "intergrax/runtime/execution/synthetic/ak_two_callsites.py"
        return (
            _synthetic_mint_surface(
                surface_id="ak::rewrite_a",
                module_path=module,
                symbol="rewrite_a",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence + (shared_scope, shared_mech, root_decision),
            ),
            _synthetic_mint_surface(
                surface_id="ak::rewrite_b",
                module_path=module,
                symbol="rewrite_b",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence + (shared_scope, shared_mech, root_decision),
            ),
        )
    if probe_id == SyntheticProbeId.AL_SAME_OWNER_DIFFERENT_DECISIONS:
        module = "intergrax/runtime/execution/synthetic/al_dual_decision.py"
        return (
            _synthetic_mint_surface(
                surface_id="al::select_root",
                module_path=module,
                symbol="select_root",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="STATE_RECOVERY",
                evidence_refs=mint_evidence + (shared_scope, root_decision),
            ),
            _synthetic_mint_surface(
                surface_id="al::restore_recovery",
                module_path="intergrax/runtime/recovery/synthetic/al_restore.py",
                symbol="restore_recovery",
                domain=TenantDomain.RECOVERY,
                semantic_owner="STATE_RECOVERY",
                evidence_refs=mint_evidence + (shared_scope, recovery_decision),
            ),
        )
    if probe_id == SyntheticProbeId.AM_COMPETING_OWNERS_SAME_DECISION:
        return (
            _synthetic_mint_surface(
                surface_id="am::owner_a",
                module_path="intergrax/runtime/execution/synthetic/am_owner_a.py",
                symbol="pick_tenant",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence + (shared_scope, root_decision),
            ),
            _synthetic_mint_surface(
                surface_id="am::owner_b",
                module_path="intergrax/governance/synthetic/am_owner_b.py",
                symbol="pick_tenant",
                domain=TenantDomain.GOVERNANCE,
                semantic_owner="GOVERNANCE",
                evidence_refs=mint_evidence + (shared_scope, root_decision),
            ),
        )
    if probe_id == SyntheticProbeId.AN_PLATFORM_CORE_UNRESOLVED:
        return (
            _synthetic_mint_surface(
                surface_id="an::unmapped",
                module_path="intergrax/platform/synthetic/an_unresolved.py",
                symbol="rewrite_unmapped",
                domain=TenantDomain.PLATFORM_GLOBAL,
                semantic_owner="PLATFORM_CORE",
                evidence_refs=mint_evidence + (root_decision,),
            ),
        )
    if probe_id == SyntheticProbeId.AO_OWNER_CONFIRMED:
        return (
            _synthetic_mint_surface(
                surface_id="ao::confirmed",
                module_path="intergrax/runtime/execution/synthetic/ao_confirmed.py",
                symbol="admit_tenant",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence
                + (
                    root_decision,
                    "TENANT-X-P0:owner:confirmed:EXECUTION",
                    "TENANT-X-P0:canonical_anchor_profile",
                ),
            ),
        )
    if probe_id == SyntheticProbeId.AP_SAME_DECISION_DOMAIN_FALLBACK:
        return (
            _synthetic_mint_surface(
                surface_id="ap::runtime",
                module_path="intergrax/runtime/execution/synthetic/ap_runtime.py",
                symbol="pick_tenant",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence + (root_decision,),
            ),
            _synthetic_mint_surface(
                surface_id="ap::governance",
                module_path="intergrax/governance/synthetic/ap_governance.py",
                symbol="pick_tenant",
                domain=TenantDomain.GOVERNANCE,
                semantic_owner="GOVERNANCE",
                evidence_refs=mint_evidence + (root_decision,),
            ),
        )
    if probe_id == SyntheticProbeId.AQ_EXPLICIT_DIFFERENT_SCOPES:
        scope_a = "TENANT-X-P0:authority_scope:root_execution_tenant"
        scope_b = "TENANT-X-P0:authority_scope:governance_policy_tenant"
        return (
            _synthetic_mint_surface(
                surface_id="aq::runtime",
                module_path="intergrax/runtime/execution/synthetic/aq_runtime.py",
                symbol="pick_tenant",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence + (scope_a, root_decision),
            ),
            _synthetic_mint_surface(
                surface_id="aq::governance",
                module_path="intergrax/governance/synthetic/aq_governance.py",
                symbol="pick_tenant",
                domain=TenantDomain.GOVERNANCE,
                semantic_owner="GOVERNANCE",
                evidence_refs=mint_evidence + (scope_b, root_decision),
            ),
        )
    if probe_id == SyntheticProbeId.AR_EXPLICIT_SAME_SCOPE_COMPETING:
        explicit_shared = "TENANT-X-P0:authority_scope:root_execution_tenant"
        return (
            _synthetic_mint_surface(
                surface_id="ar::owner_a",
                module_path="intergrax/runtime/execution/synthetic/ar_owner_a.py",
                symbol="pick_tenant",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence + (explicit_shared, root_decision),
            ),
            _synthetic_mint_surface(
                surface_id="ar::owner_b",
                module_path="intergrax/governance/synthetic/ar_owner_b.py",
                symbol="pick_tenant",
                domain=TenantDomain.GOVERNANCE,
                semantic_owner="GOVERNANCE",
                evidence_refs=mint_evidence + (explicit_shared, root_decision),
            ),
        )
    if probe_id == SyntheticProbeId.AS_SAME_MECHANISM_CALL_SITES:
        module = "intergrax/runtime/execution/synthetic/as_two_callsites.py"
        explicit_shared = "TENANT-X-P0:authority_scope:root_execution_tenant"
        return (
            _synthetic_mint_surface(
                surface_id="as::rewrite_a",
                module_path=module,
                symbol="rewrite_a",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence
                + (explicit_shared, shared_mech, root_decision),
            ),
            _synthetic_mint_surface(
                surface_id="as::rewrite_b",
                module_path=module,
                symbol="rewrite_b",
                domain=TenantDomain.RUNTIME_REQUEST_EXECUTION,
                semantic_owner="EXECUTION",
                evidence_refs=mint_evidence
                + (explicit_shared, shared_mech, root_decision),
            ),
        )
    raise AssertionError(f"unknown authority mechanism probe {probe_id}")


AUTHORITY_R3R1_PROBE_SCENARIOS: tuple[AuthorityMechanismProbeScenario, ...] = (
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AK_ONE_OWNER_TWO_CALL_SITES,
        description="one owner, two call-sites, same sanctioned mechanism",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AL_SAME_OWNER_DIFFERENT_DECISIONS,
        description="same owner, distinct decision kinds",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AM_COMPETING_OWNERS_SAME_DECISION,
        description="competing owners for same decision/scope",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AN_PLATFORM_CORE_UNRESOLVED,
        description="PLATFORM_CORE maps to owner unresolved, not unowned",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AO_OWNER_CONFIRMED,
        description="explicit owner confirmation evidence",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AP_SAME_DECISION_DOMAIN_FALLBACK,
        description="same decision, different domains, domain-fallback scope only",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AQ_EXPLICIT_DIFFERENT_SCOPES,
        description="same decision, explicit different authority scopes",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AR_EXPLICIT_SAME_SCOPE_COMPETING,
        description="same decision, same explicit scope, competing owners",
    ),
    AuthorityMechanismProbeScenario(
        probe_id=SyntheticProbeId.AS_SAME_MECHANISM_CALL_SITES,
        description="same owner, explicit scope, multiple call-sites",
    ),
)


AUTHORITY_R3_PROBE_SCENARIOS: tuple[ClassifierProbeScenario, ...] = (
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AA_VALIDATOR_ONLY,
        module_path="synthetic/authority_probes/aa_validator.py",
        source=_SOURCE_AA,
        surface_selector="require_tenant_id",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AB_PARENT_PROPAGATION,
        module_path="synthetic/authority_probes/ab_parent_propagation.py",
        source=_SOURCE_AB,
        surface_selector="build_child",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AC_ILLEGAL_REWRITE,
        module_path="synthetic/authority_probes/ac_illegal_rewrite.py",
        source=_SOURCE_AC,
        surface_selector="build_bad",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AD_SCOPE_NARROWING,
        module_path="synthetic/authority_probes/ad_narrowing.py",
        source=_SOURCE_AD,
        surface_selector="narrow_scope",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AE_SCOPE_WIDENING,
        module_path="synthetic/authority_probes/ae_widening.py",
        source=_SOURCE_AE,
        surface_selector="widen_scope",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AF_SILENT_DEFAULT,
        module_path="synthetic/authority_probes/af_silent_default.py",
        source=_SOURCE_AF,
        surface_selector="resolve_tenant",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AG_CREDENTIAL_SAME,
        module_path="intergrax/integrations/credentials/synthetic/ag_same.py",
        source=_SOURCE_AG,
        surface_selector="use_credential",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AH_CREDENTIAL_MISMATCH,
        module_path="intergrax/integrations/credentials/synthetic/ah_mismatch.py",
        source=_SOURCE_AH,
        surface_selector="bad_credential",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AI_CHECKPOINT_SAME,
        module_path="intergrax/runtime/checkpoint/synthetic/ai_same.py",
        source=_SOURCE_AI,
        surface_selector="restore",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.AJ_CHECKPOINT_SUBSTITUTION,
        module_path="intergrax/runtime/checkpoint/synthetic/aj_substitution.py",
        source=_SOURCE_AJ,
        surface_selector="resume",
    ),
)

CARRIER_R2_PROBE_SCENARIOS: tuple[ClassifierProbeScenario, ...] = (
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Z1_PASSIVE_DTO,
        module_path="intergrax/runtime/schema/z1_passive_evidence_fixture.py",
        source=_SOURCE_Z1,
        surface_selector="EvidenceRecord",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Z2_EXECUTION_BOUNDARY,
        module_path="intergrax/runtime/execution/z2_admission_fixture.py",
        source=_SOURCE_Z2,
        surface_selector="RuntimeAdmissionEnvelope",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Z3_STATE_DTO,
        module_path="intergrax/runtime/schema/z3_state_fixture.py",
        source=_SOURCE_Z3,
        surface_selector="CheckpointRow",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Z4_APPLICATION_SCHEMA,
        module_path="applications/demo/z4_app_schema_fixture.py",
        source=_SOURCE_Z4,
        surface_selector="AppTenantPayload",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Z5_EVIDENCE_EVENT,
        module_path="intergrax/runtime/events/z5_event_fixture.py",
        source=_SOURCE_Z5,
        surface_selector="RuntimeEventRecord",
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Z6_FAMILY_UNKNOWN,
        module_path="synthetic/carrier_family_unknown/z6_unclassified.py",
        source=_SOURCE_Z6,
        surface_selector="UnclassifiedCarrier",
    ),
)


def _select_record(
    scenario: ClassifierProbeScenario,
    *,
    prefer_body_authority: bool = False,
) -> TenantSurfaceRecord:
    records = classify_source_fixture(scenario.module_path, scenario.source)
    if scenario.surface_selector == "ambient_default_field":
        matches = [r for r in records if "ambient_default" in r.surface_id]
        if not matches:
            raise AssertionError(f"no ambient default surface in fixture {scenario.probe_id}")
        return matches[0]
    matches = [
        record
        for record in records
        if record.symbol == scenario.surface_selector or scenario.surface_selector in record.surface_id
    ]
    if matches:
        prefs = (
            ("function.body.authority", "function.tenant_kwarg.site", "function.param.tenant")
            if prefer_body_authority
            else ("function.param.tenant", "function.tenant_kwarg.site", "function.body.authority")
        )
        for pref in prefs:
            for record in matches:
                if pref in record.surface_id:
                    return record
        return matches[0]
    raise AssertionError(f"surface {scenario.surface_selector} not found for probe {scenario.probe_id}")


def classify_synthetic_probe(scenario: SyntheticAdversarialScenario) -> TenantPropagationRole:
    probe = scenario.probe_id
    if probe == SyntheticProbeId.I_LEGAL_CROSS_TENANT_CONTRACT:
        return TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY
    if probe == SyntheticProbeId.J_GLOBAL_HEALTH_METRIC:
        return TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE
    if probe == SyntheticProbeId.C_MISSING_TENANT_DEFAULT:
        return TenantPropagationRole.IDENTITY_DERIVATION
    if probe == SyntheticProbeId.H_PLUGIN_REWRITES_TENANT:
        return TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION
    if probe in {
        SyntheticProbeId.A_CROSS_TENANT_STATE_KEY,
        SyntheticProbeId.G_CROSS_TENANT_CHECKPOINT_RECOVERY,
    }:
        return TenantPropagationRole.TENANT_SCOPED_WRITE
    if probe == SyntheticProbeId.B_CHILD_WIDENS_TENANT:
        return TenantPropagationRole.TENANT_SCOPED_CHILD_OPERATION
    if probe == SyntheticProbeId.D_CROSS_TENANT_PROVIDER_CONFIG:
        return TenantPropagationRole.TENANT_SCOPED_CONFIGURATION
    if probe == SyntheticProbeId.E_CROSS_TENANT_CREDENTIAL:
        return TenantPropagationRole.TENANT_SCOPED_CREDENTIAL_REFERENCE
    if probe == SyntheticProbeId.F_CROSS_TENANT_EVIDENCE:
        return TenantPropagationRole.TENANT_SCOPED_EVIDENCE
    return TenantPropagationRole.UNKNOWN


def synthetic_probe_fail_closed(scenario: SyntheticAdversarialScenario) -> TenantFailClosedDisposition:
    role = classify_synthetic_probe(scenario)
    if role == TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE:
        return TenantFailClosedDisposition.NON_TENANT_SURFACE
    if role == TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY:
        return TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH
    if scenario.probe_id == SyntheticProbeId.C_MISSING_TENANT_DEFAULT:
        return TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION
    return TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH


def evaluate_synthetic_probe(scenario: SyntheticAdversarialScenario) -> bool:
    """Return True when classification matches P0 expected PASS/FAIL bucket."""
    role = classify_synthetic_probe(scenario)
    fail_closed = synthetic_probe_fail_closed(scenario)
    if scenario.expected_verdict == AdversarialProbeVerdict.EXPECT_PASS_CLASSIFICATION:
        return role in {
            TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY,
            TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE,
        }
    if role == TenantPropagationRole.UNKNOWN:
        return False
    if scenario.probe_id == SyntheticProbeId.C_MISSING_TENANT_DEFAULT:
        return fail_closed == TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION
    return fail_closed in {
        TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
        TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION,
    }


def evaluate_classifier_probe(scenario: ClassifierProbeScenario) -> bool:
    record = _select_record(scenario)
    probe = scenario.probe_id
    if probe == SyntheticProbeId.K_UNKNOWN_TENANT_FIELD:
        return (
            record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
            and record.identity_source == TenantIdentitySource.UNKNOWN
            and record.authority_role == TenantAuthorityRole.NOT_APPLICABLE
        )
    if probe == SyntheticProbeId.L_FUNCTION_PARAM_ONLY:
        return (
            record.identity_source == TenantIdentitySource.EXPLICIT_PARAMETER
            and record.fail_closed_behavior == TenantFailClosedDisposition.UNKNOWN
            and record.persistence_role == TenantPersistenceRole.NOT_APPLICABLE
        )
    if probe == SyntheticProbeId.M_SILENT_TENANT_DEFAULT:
        return record.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER
    if probe == SyntheticProbeId.N_HARMLESS_GLOBAL_METADATA:
        return record.blocker_class == TenantBlockerClass.NONE and record.tenant_required is False
    if probe == SyntheticProbeId.O_CREDENTIALS_PATH_ONLY:
        return record.propagation_role == TenantPropagationRole.UNKNOWN
    if probe == SyntheticProbeId.P_STORAGE_IGNORES_TENANT:
        return (
            record.persistence_role == TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE
            and record.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER
        )
    if probe == SyntheticProbeId.Q_ACTUAL_TENANT_PREDICATE:
        return record.persistence_role == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE
    if probe == SyntheticProbeId.R_VALIDATOR_NOT_ON_LOOKUP:
        return (
            record.persistence_role == TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE
            and "lookup_without_validator" in "".join(record.evidence_refs)
        )
    if probe == SyntheticProbeId.S_VALIDATOR_BEFORE_LOOKUP:
        return record.persistence_role == TenantPersistenceRole.TENANT_VALIDATED_BEFORE_LOOKUP
    if probe == SyntheticProbeId.T_UNRELATED_DICT_SUBSCRIPT:
        return record.blocker_class != TenantBlockerClass.IN_SCOPE_BLOCKER
    if probe == SyntheticProbeId.U_GLOBAL_REFERENCE_SQL:
        return record.blocker_class == TenantBlockerClass.NONE
    if probe == SyntheticProbeId.V_TENANT_STORE_MISSING_KEY:
        return (
            record.persistence_role == TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE
            and record.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER
        )
    if probe == SyntheticProbeId.W_TENANT_STORE_TENANT_KEY:
        return record.persistence_role == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE
    if probe == SyntheticProbeId.X_VALIDATOR_AFTER_LOOKUP:
        return record.persistence_role != TenantPersistenceRole.TENANT_VALIDATED_BEFORE_LOOKUP
    if probe == SyntheticProbeId.Y_VALIDATOR_BEFORE_TENANT_KEY:
        return record.persistence_role == TenantPersistenceRole.TENANT_VALIDATED_BEFORE_LOOKUP
    return False


def evaluate_authority_r3r1_probe(scenario: AuthorityMechanismProbeScenario) -> bool:
    surfaces = _authority_r3r1_probe_surfaces(scenario.probe_id)
    mechanisms = build_authority_mechanisms(surfaces)
    duplicate_review = classify_authority_duplicate_mechanisms(mechanisms)
    probe = scenario.probe_id
    if probe == SyntheticProbeId.AK_ONE_OWNER_TWO_CALL_SITES:
        return (
            len(surfaces) == 2
            and len(mechanisms) == 1
            and duplicate_review.confirmed_duplicate_mechanism_ids == ()
        )
    if probe == SyntheticProbeId.AL_SAME_OWNER_DIFFERENT_DECISIONS:
        return len(mechanisms) == 2 and duplicate_review.confirmed_duplicate_mechanism_ids == ()
    if probe == SyntheticProbeId.AM_COMPETING_OWNERS_SAME_DECISION:
        return len(mechanisms) == 2 and len(duplicate_review.confirmed_duplicate_mechanism_ids) >= 1
    if probe == SyntheticProbeId.AN_PLATFORM_CORE_UNRESOLVED:
        metrics = build_authority_review_metrics(surfaces)
        disposition = resolve_owner_disposition(surfaces[0])
        return (
            disposition == TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED
            and metrics.unowned_authority_mechanisms == ()
        )
    if probe == SyntheticProbeId.AO_OWNER_CONFIRMED:
        return resolve_owner_disposition(surfaces[0]) == TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
    if probe == SyntheticProbeId.AP_SAME_DECISION_DOMAIN_FALLBACK:
        return (
            len(mechanisms) == 2
            and duplicate_review.confirmed_duplicate_mechanism_ids == ()
            and duplicate_review.proven_no_duplicate_mechanism_ids == ()
            and len(duplicate_review.unresolved_duplicate_mechanism_ids) >= 1
        )
    if probe == SyntheticProbeId.AQ_EXPLICIT_DIFFERENT_SCOPES:
        return (
            len(mechanisms) == 2
            and duplicate_review.confirmed_duplicate_mechanism_ids == ()
            and len(duplicate_review.proven_no_duplicate_mechanism_ids) == 2
            and duplicate_review.unresolved_duplicate_mechanism_ids == ()
        )
    if probe == SyntheticProbeId.AR_EXPLICIT_SAME_SCOPE_COMPETING:
        return len(mechanisms) == 2 and len(duplicate_review.confirmed_duplicate_mechanism_ids) >= 1
    if probe == SyntheticProbeId.AS_SAME_MECHANISM_CALL_SITES:
        return (
            len(surfaces) == 2
            and len(mechanisms) == 1
            and duplicate_review.confirmed_duplicate_mechanism_ids == ()
        )
    return False


def evaluate_authority_r3_probe(scenario: ClassifierProbeScenario) -> bool:
    record = _select_record(scenario, prefer_body_authority=True)
    probe = scenario.probe_id
    if probe == SyntheticProbeId.AA_VALIDATOR_ONLY:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.VALIDATOR
    if probe == SyntheticProbeId.AB_PARENT_PROPAGATION:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.DERIVER
    if probe == SyntheticProbeId.AC_ILLEGAL_REWRITE:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY
    if probe == SyntheticProbeId.AD_SCOPE_NARROWING:
        return record.authority_semantic_role in {
            TenantAuthoritySemanticRole.VALIDATOR,
            TenantAuthoritySemanticRole.DERIVER,
        }
    if probe == SyntheticProbeId.AE_SCOPE_WIDENING:
        return record.authority_semantic_role in {
            TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY,
            TenantAuthoritySemanticRole.UNKNOWN,
        }
    if probe == SyntheticProbeId.AF_SILENT_DEFAULT:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY
    if probe == SyntheticProbeId.AG_CREDENTIAL_SAME:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.VALIDATOR
    if probe == SyntheticProbeId.AH_CREDENTIAL_MISMATCH:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.VALIDATOR
    if probe == SyntheticProbeId.AI_CHECKPOINT_SAME:
        return record.authority_semantic_role == TenantAuthoritySemanticRole.DERIVER
    if probe == SyntheticProbeId.AJ_CHECKPOINT_SUBSTITUTION:
        return record.authority_semantic_role in {
            TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY,
            TenantAuthoritySemanticRole.UNKNOWN,
        }
    return False


def evaluate_carrier_r2_probe(scenario: ClassifierProbeScenario) -> bool:
    record = _select_record(scenario)
    probe = scenario.probe_id
    if probe == SyntheticProbeId.Z1_PASSIVE_DTO:
        return (
            record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
            and record.carrier_provenance_requirement
            == TenantCarrierProvenanceRequirement.PASSIVE_CARRIER_ONLY
            and record.identity_source == TenantIdentitySource.UNKNOWN
            and record.authority_role == TenantAuthorityRole.NOT_APPLICABLE
        )
    if probe == SyntheticProbeId.Z2_EXECUTION_BOUNDARY:
        return (
            record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
            and record.carrier_family == TenantCarrierFamily.RUNTIME_EXECUTION
            and record.carrier_provenance_requirement
            == TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_BOUNDARY
            and record.authority_role == TenantAuthorityRole.NOT_APPLICABLE
        )
    if probe == SyntheticProbeId.Z3_STATE_DTO:
        return (
            record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
            and record.carrier_family == TenantCarrierFamily.STATE_DURABLE
            and record.carrier_provenance_requirement
            == TenantCarrierProvenanceRequirement.PASSIVE_CARRIER_ONLY
        )
    if probe == SyntheticProbeId.Z4_APPLICATION_SCHEMA:
        return (
            record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
            and record.carrier_family == TenantCarrierFamily.APPLICATION_BOUNDARY
            and record.authority_role != TenantAuthorityRole.TENANT_SCOPE_AUTHORITY
        )
    if probe == SyntheticProbeId.Z5_EVIDENCE_EVENT:
        return (
            record.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
            and record.carrier_family == TenantCarrierFamily.EVIDENCE_TRACE
            and record.carrier_provenance_requirement
            == TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_USE_SITE
        )
    if probe == SyntheticProbeId.Z6_FAMILY_UNKNOWN:
        return record.carrier_family == TenantCarrierFamily.UNKNOWN
    return False
