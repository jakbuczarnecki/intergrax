# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Reference local analyzer — rule-based, no LLM, no execution side effects (W6-C)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from intergrax.contracts.runtime_intelligence.context import (
    RuntimeIntelligenceContext,
    RuntimeIntelligenceFactKind,
)
from intergrax.contracts.runtime_intelligence.evidence import (
    IntelligenceEvidence,
    IntelligenceEvidenceSourceKind,
)
from intergrax.contracts.runtime_intelligence.recommendation import (
    IntelligenceRecommendation,
    IntelligenceRecommendationKind,
)
from intergrax.contracts.runtime_intelligence.result import RuntimeIntelligenceResult
from intergrax.runtime.runtime_intelligence.context_builder import intelligence_signal_from_fact_ref
from intergrax.runtime.runtime_intelligence.runtime_facts import RuntimeIntelligenceSignalKind

_ANALYZER_ID = "runtime_intelligence.deterministic"
_ANALYZER_VERSION = "1.0.0"
_SIGNAL_THRESHOLD = 0.5

_KIND_TO_RECOMMENDATION: dict[RuntimeIntelligenceSignalKind, tuple[str, IntelligenceRecommendationKind]] = {
    RuntimeIntelligenceSignalKind.EXECUTION_INSTABILITY: (
        "Review recent phase transitions and dependency timing for this run.",
        IntelligenceRecommendationKind.RUNTIME_DIAGNOSTIC_HINT,
    ),
    RuntimeIntelligenceSignalKind.REPEATED_FAILURES: (
        "Correlate terminal outcomes before scheduling another attempt.",
        IntelligenceRecommendationKind.EXECUTION_INSIGHT,
    ),
    RuntimeIntelligenceSignalKind.RETRY_PRESSURE: (
        "Inspect retry budget consumption and backoff policy fit.",
        IntelligenceRecommendationKind.ADAPTIVE_POLICY_HINT,
    ),
    RuntimeIntelligenceSignalKind.RECOVERY_SIGNAL: (
        "Audit recovery admission decisions against checkpoint lineage.",
        IntelligenceRecommendationKind.DECISION_AUDIT_NOTE,
    ),
    RuntimeIntelligenceSignalKind.RESOURCE_PRESSURE: (
        "Check admission and capacity signals before increasing concurrency.",
        IntelligenceRecommendationKind.ADAPTIVE_POLICY_HINT,
    ),
}


def _stable_token(*parts: str) -> str:
    return sha256("|".join(parts).encode()).hexdigest()[:16]


def _primary_fact_ref(context: RuntimeIntelligenceContext) -> str:
    return context.fact_references[0].fact_ref


def _evidence_source_kind(fact_kind: RuntimeIntelligenceFactKind) -> IntelligenceEvidenceSourceKind:
    if fact_kind == RuntimeIntelligenceFactKind.CHECKPOINT:
        return IntelligenceEvidenceSourceKind.CHECKPOINT
    if fact_kind == RuntimeIntelligenceFactKind.TERMINAL:
        return IntelligenceEvidenceSourceKind.TERMINAL
    if fact_kind == RuntimeIntelligenceFactKind.LINEAGE:
        return IntelligenceEvidenceSourceKind.LINEAGE
    return IntelligenceEvidenceSourceKind.RUNTIME_EVENT


@dataclass(frozen=True, slots=True)
class DeterministicRuntimeIntelligenceAnalyzer:
    """Plugin SPI reference implementation — deterministic evidence + recommendations."""

    analyzer_id: str = _ANALYZER_ID
    analyzer_version: str = _ANALYZER_VERSION
    signal_threshold: float = _SIGNAL_THRESHOLD

    def analyze(self, context: RuntimeIntelligenceContext) -> RuntimeIntelligenceResult:
        elevated = _elevated_signals(context, self.signal_threshold)
        recovery_refs = _fact_refs_by_kind(context, RuntimeIntelligenceFactKind.RECOVERY_RECORD)
        terminal_refs = _fact_refs_by_kind(context, RuntimeIntelligenceFactKind.TERMINAL)

        evidence = _build_evidence(context, elevated, recovery_refs, terminal_refs)
        recommendations = _build_recommendations(context, elevated)
        confidence = _confidence(elevated)
        summary = _analysis_summary(elevated, recovery_refs, terminal_refs)

        return RuntimeIntelligenceResult(
            analysis_summary=summary,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            degraded=False,
        )


def _elevated_signals(
    context: RuntimeIntelligenceContext,
    threshold: float,
) -> tuple[tuple[RuntimeIntelligenceSignalKind, float, str], ...]:
    parsed: list[tuple[RuntimeIntelligenceSignalKind, float, str]] = []
    for fact in context.fact_references:
        decoded = intelligence_signal_from_fact_ref(fact.fact_ref)
        if decoded is None:
            continue
        kind, intensity = decoded
        if intensity >= threshold:
            parsed.append((kind, intensity, fact.fact_ref))
    parsed.sort(key=lambda item: (item[0].value, item[2]))
    return tuple(parsed)


def _fact_refs_by_kind(
    context: RuntimeIntelligenceContext,
    kind: RuntimeIntelligenceFactKind,
) -> tuple[str, ...]:
    refs = tuple(ref.fact_ref for ref in context.fact_references if ref.fact_kind == kind)
    return tuple(sorted(refs))


def _build_evidence(
    context: RuntimeIntelligenceContext,
    elevated: tuple[tuple[RuntimeIntelligenceSignalKind, float, str], ...],
    recovery_refs: tuple[str, ...],
    terminal_refs: tuple[str, ...],
) -> tuple[IntelligenceEvidence, ...]:
    items: list[IntelligenceEvidence] = []
    run_token = _stable_token(context.tenant_id, context.task_id, context.run_id)

    if elevated:
        for kind, intensity, fact_ref in elevated:
            items.append(
                IntelligenceEvidence(
                    evidence_id=f"ev_{run_token}_{kind.value}",
                    source_kind=IntelligenceEvidenceSourceKind.RUNTIME_EVENT,
                    source_ref=fact_ref,
                    relation="elevated_signal",
                    summary=f"{kind.value} intensity {intensity:.2f}",
                )
            )
    else:
        anchor = context.fact_references[0]
        items.append(
            IntelligenceEvidence(
                evidence_id=f"ev_{run_token}_baseline",
                source_kind=_evidence_source_kind(anchor.fact_kind),
                source_ref=anchor.fact_ref,
                relation="observed_fact",
                summary="No elevated intelligence signals; baseline fact anchor",
            )
        )

    for ref in recovery_refs:
        items.append(
            IntelligenceEvidence(
                evidence_id=f"ev_{run_token}_recovery_{_stable_token(ref)}",
                source_kind=IntelligenceEvidenceSourceKind.RUNTIME_EVENT,
                source_ref=ref,
                relation="recovery_record",
                summary="Recovery record present in context",
            )
        )

    if len(terminal_refs) >= 2:
        items.append(
            IntelligenceEvidence(
                evidence_id=f"ev_{run_token}_terminal_pattern",
                source_kind=IntelligenceEvidenceSourceKind.TERMINAL,
                source_ref=terminal_refs[0],
                relation="repeated_terminal",
                summary=f"{len(terminal_refs)} terminal fact pointers observed",
            )
        )

    return tuple(items)


def _build_recommendations(
    context: RuntimeIntelligenceContext,
    elevated: tuple[tuple[RuntimeIntelligenceSignalKind, float, str], ...],
) -> tuple[IntelligenceRecommendation, ...]:
    run_token = _stable_token(context.tenant_id, context.task_id, context.run_id)
    recs: list[IntelligenceRecommendation] = []
    for kind, intensity, _fact_ref in elevated:
        summary, rec_kind = _KIND_TO_RECOMMENDATION[kind]
        priority = "HIGH" if intensity >= 0.75 else "NORMAL"
        recs.append(
            IntelligenceRecommendation(
                recommendation_id=f"rec_{run_token}_{kind.value}",
                kind=rec_kind,
                summary=summary,
                rationale=f"Deterministic rule: {kind.value} >= {intensity:.2f}",
                priority_label=priority,
            )
        )
    if not recs:
        recs.append(
            IntelligenceRecommendation(
                recommendation_id=f"rec_{run_token}_stable",
                kind=IntelligenceRecommendationKind.EXECUTION_INSIGHT,
                summary="No elevated runtime intelligence signals; continue standard investigation.",
                rationale="All projected signals below threshold",
                priority_label="LOW",
            )
        )
    return tuple(recs)


def _confidence(
    elevated: tuple[tuple[RuntimeIntelligenceSignalKind, float, str], ...],
) -> float:
    if not elevated:
        return 0.55
    peak = max(intensity for _, intensity, _ in elevated)
    return min(0.95, 0.5 + peak * 0.45)


def _analysis_summary(
    elevated: tuple[tuple[RuntimeIntelligenceSignalKind, float, str], ...],
    recovery_refs: tuple[str, ...],
    terminal_refs: tuple[str, ...],
) -> str:
    if elevated:
        kinds = ", ".join(kind.value for kind, _, _ in elevated)
        return f"Elevated runtime intelligence signals: {kinds}"
    if recovery_refs:
        return "Recovery records observed without elevated signal projections"
    if len(terminal_refs) >= 2:
        return "Multiple terminal fact pointers suggest repeated failure pattern"
    return "Runtime intelligence baseline — no elevated deterministic signals"


__all__ = ["DeterministicRuntimeIntelligenceAnalyzer"]
