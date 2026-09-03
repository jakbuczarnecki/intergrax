# © Artur Czarnecki. All rights reserved.

"""Typed immutable scale profiles for DIAG-FUNCTIONAL-SCALE-S1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.knowledge.contracts.validation import JsonObject, JsonValue


class FunctionalDiagnosticsScaleProfileName(StrEnum):
    SMOKE = "SMOKE"
    STANDARD = "STANDARD"
    STRESS = "STRESS"


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosticsScaleProfile:
    """Immutable load envelope for functional diagnostics scale qualification."""

    name: FunctionalDiagnosticsScaleProfileName
    seed: int
    tenant_count: int
    execution_count_per_tenant: int
    typical_evidence_per_execution: int
    heavy_execution_count_per_tenant: int
    heavy_evidence_per_execution: int
    writer_concurrency: int
    reader_concurrency: int
    page_size: int
    document_store_query_page_limit: int
    analyzer_sample_executions_per_tenant: int
    scale_curve_probe_evidence_count: int
    tenant_namespace: str = "diag-s1-tenant"

    def total_executions(self) -> int:
        return self.tenant_count * self.execution_count_per_tenant

    def expected_typical_evidence(self) -> int:
        typical_executions = (
            self.execution_count_per_tenant - self.heavy_execution_count_per_tenant
        )
        return (
            self.tenant_count
            * typical_executions
            * self.typical_evidence_per_execution
        )

    def expected_heavy_evidence(self) -> int:
        return (
            self.tenant_count
            * self.heavy_execution_count_per_tenant
            * self.heavy_evidence_per_execution
        )

    def expected_total_evidence(self) -> int:
        return self.expected_typical_evidence() + self.expected_heavy_evidence()

    def to_json_dict(self) -> JsonObject:
        return {
            "name": self.name.value,
            "seed": self.seed,
            "tenant_count": self.tenant_count,
            "execution_count_per_tenant": self.execution_count_per_tenant,
            "typical_evidence_per_execution": self.typical_evidence_per_execution,
            "heavy_execution_count_per_tenant": self.heavy_execution_count_per_tenant,
            "heavy_evidence_per_execution": self.heavy_evidence_per_execution,
            "writer_concurrency": self.writer_concurrency,
            "reader_concurrency": self.reader_concurrency,
            "page_size": self.page_size,
            "document_store_query_page_limit": self.document_store_query_page_limit,
            "analyzer_sample_executions_per_tenant": self.analyzer_sample_executions_per_tenant,
            "scale_curve_probe_evidence_count": self.scale_curve_probe_evidence_count,
            "tenant_namespace": self.tenant_namespace,
            "total_executions": self.total_executions(),
            "expected_total_evidence": self.expected_total_evidence(),
        }

    @classmethod
    def from_json_mapping(cls, payload: object) -> FunctionalDiagnosticsScaleProfile:
        if not isinstance(payload, dict):
            raise ValueError("scale_profile_invalid")
        return cls(
            name=FunctionalDiagnosticsScaleProfileName(str(payload["name"])),
            seed=int(payload["seed"]),
            tenant_count=int(payload["tenant_count"]),
            execution_count_per_tenant=int(payload["execution_count_per_tenant"]),
            typical_evidence_per_execution=int(payload["typical_evidence_per_execution"]),
            heavy_execution_count_per_tenant=int(
                payload["heavy_execution_count_per_tenant"],
            ),
            heavy_evidence_per_execution=int(payload["heavy_evidence_per_execution"]),
            writer_concurrency=int(payload["writer_concurrency"]),
            reader_concurrency=int(payload["reader_concurrency"]),
            page_size=int(payload["page_size"]),
            document_store_query_page_limit=int(
                payload["document_store_query_page_limit"],
            ),
            analyzer_sample_executions_per_tenant=int(
                payload["analyzer_sample_executions_per_tenant"],
            ),
            scale_curve_probe_evidence_count=int(
                payload["scale_curve_probe_evidence_count"],
            ),
            tenant_namespace=str(payload.get("tenant_namespace", "diag-s1-tenant")),
        )


_PROFILE_SMOKE = FunctionalDiagnosticsScaleProfile(
    name=FunctionalDiagnosticsScaleProfileName.SMOKE,
    seed=20260903,
    tenant_count=2,
    execution_count_per_tenant=10,
    typical_evidence_per_execution=12,
    heavy_execution_count_per_tenant=1,
    heavy_evidence_per_execution=60,
    writer_concurrency=2,
    reader_concurrency=2,
    page_size=10,
    document_store_query_page_limit=25,
    analyzer_sample_executions_per_tenant=1,
    scale_curve_probe_evidence_count=12,
)

_PROFILE_STANDARD = FunctionalDiagnosticsScaleProfile(
    name=FunctionalDiagnosticsScaleProfileName.STANDARD,
    seed=20260903,
    tenant_count=12,
    execution_count_per_tenant=120,
    typical_evidence_per_execution=18,
    heavy_execution_count_per_tenant=2,
    heavy_evidence_per_execution=240,
    writer_concurrency=4,
    reader_concurrency=4,
    page_size=25,
    document_store_query_page_limit=100,
    analyzer_sample_executions_per_tenant=2,
    scale_curve_probe_evidence_count=20,
)

_PROFILE_STRESS = FunctionalDiagnosticsScaleProfile(
    name=FunctionalDiagnosticsScaleProfileName.STRESS,
    seed=20260903,
    tenant_count=16,
    execution_count_per_tenant=200,
    typical_evidence_per_execution=24,
    heavy_execution_count_per_tenant=3,
    heavy_evidence_per_execution=400,
    writer_concurrency=6,
    reader_concurrency=6,
    page_size=25,
    document_store_query_page_limit=100,
    analyzer_sample_executions_per_tenant=3,
    scale_curve_probe_evidence_count=20,
)

_CANONICAL_S1_PROFILE = _PROFILE_STANDARD

_SCALE_PROFILES: dict[FunctionalDiagnosticsScaleProfileName, FunctionalDiagnosticsScaleProfile] = {
    FunctionalDiagnosticsScaleProfileName.SMOKE: _PROFILE_SMOKE,
    FunctionalDiagnosticsScaleProfileName.STANDARD: _PROFILE_STANDARD,
    FunctionalDiagnosticsScaleProfileName.STRESS: _PROFILE_STRESS,
}


def resolve_scale_profile(
    name: FunctionalDiagnosticsScaleProfileName | str,
) -> FunctionalDiagnosticsScaleProfile:
    resolved = (
        name
        if isinstance(name, FunctionalDiagnosticsScaleProfileName)
        else FunctionalDiagnosticsScaleProfileName(str(name).upper())
    )
    return _SCALE_PROFILES[resolved]


def canonical_s1_profile() -> FunctionalDiagnosticsScaleProfile:
    return _CANONICAL_S1_PROFILE


def scale_curve_profiles(
    probe_evidence_count: int,
) -> tuple[FunctionalDiagnosticsScaleProfile, ...]:
    """Small/medium/large cardinalities with fixed per-execution evidence size."""
    base_seed = _CANONICAL_S1_PROFILE.seed
    return (
        FunctionalDiagnosticsScaleProfile(
            name=FunctionalDiagnosticsScaleProfileName.SMOKE,
            seed=base_seed,
            tenant_count=2,
            execution_count_per_tenant=40,
            typical_evidence_per_execution=probe_evidence_count,
            heavy_execution_count_per_tenant=0,
            heavy_evidence_per_execution=0,
            writer_concurrency=1,
            reader_concurrency=1,
            page_size=10,
            document_store_query_page_limit=50,
            analyzer_sample_executions_per_tenant=0,
            scale_curve_probe_evidence_count=probe_evidence_count,
            tenant_namespace="diag-s1-curve-small",
        ),
        FunctionalDiagnosticsScaleProfile(
            name=FunctionalDiagnosticsScaleProfileName.STANDARD,
            seed=base_seed,
            tenant_count=6,
            execution_count_per_tenant=80,
            typical_evidence_per_execution=probe_evidence_count,
            heavy_execution_count_per_tenant=0,
            heavy_evidence_per_execution=0,
            writer_concurrency=1,
            reader_concurrency=1,
            page_size=10,
            document_store_query_page_limit=50,
            analyzer_sample_executions_per_tenant=0,
            scale_curve_probe_evidence_count=probe_evidence_count,
            tenant_namespace="diag-s1-curve-medium",
        ),
        FunctionalDiagnosticsScaleProfile(
            name=FunctionalDiagnosticsScaleProfileName.STRESS,
            seed=base_seed,
            tenant_count=12,
            execution_count_per_tenant=120,
            typical_evidence_per_execution=probe_evidence_count,
            heavy_execution_count_per_tenant=0,
            heavy_evidence_per_execution=0,
            writer_concurrency=1,
            reader_concurrency=1,
            page_size=10,
            document_store_query_page_limit=50,
            analyzer_sample_executions_per_tenant=0,
            scale_curve_probe_evidence_count=probe_evidence_count,
            tenant_namespace="diag-s1-curve-large",
        ),
    )


def profile_to_json_value(profile: FunctionalDiagnosticsScaleProfile) -> JsonValue:
    return profile.to_json_dict()


__all__ = [
    "FunctionalDiagnosticsScaleProfile",
    "FunctionalDiagnosticsScaleProfileName",
    "canonical_s1_profile",
    "profile_to_json_value",
    "resolve_scale_profile",
    "scale_curve_profiles",
]
