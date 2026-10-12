# © Artur Czarnecki. All rights reserved.

"""In-memory adversarial probes (TENANT-X-P0 §14–15) — source fixtures only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.tenant_x._classifiers import classify_source_fixture
from tests.qualification.tenant_x._types import (
    AdversarialProbeVerdict,
    TenantBlockerClass,
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


def _select_record(scenario: ClassifierProbeScenario) -> TenantSurfaceRecord:
    records = classify_source_fixture(scenario.module_path, scenario.source)
    if scenario.surface_selector == "ambient_default_field":
        matches = [r for r in records if "ambient_default" in r.surface_id]
        if not matches:
            raise AssertionError(f"no ambient default surface in fixture {scenario.probe_id}")
        return matches[0]
    for record in records:
        if record.symbol == scenario.surface_selector:
            return record
        if scenario.surface_selector in record.surface_id:
            return record
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
        return record.propagation_role == TenantPropagationRole.UNKNOWN
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
