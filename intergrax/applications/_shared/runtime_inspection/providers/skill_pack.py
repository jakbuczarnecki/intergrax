# © Artur Czarnecki. All rights reserved.

"""Resolved skill pack inspection provider (P1.10)."""

from __future__ import annotations

from intergrax.applications.contracts.runtime_inspection.explanation import (
    InspectionExplanation,
    InspectionProvenanceRef,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    InspectionExtensionEvidence,
    InspectionProviderContribution,
    RuntimeInspectionProvider,
)
from intergrax.applications.contracts.runtime_inspection.scope import InspectionScope
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.skills.contribution_provenance import SkillContributionKind
from intergrax.skills.execution_binding import SkillExecutionPinningStore


_PROVIDER_ID = "skill_pack_binding"


def _ref_facts(ref: object) -> tuple[str, ...]:
    from intergrax.skills.core.version_binding import ResolvedSkillRef

    if not isinstance(ref, ResolvedSkillRef):
        return ()
    return (
        f"skill_id={ref.skill_id}",
        f"version={ref.version}",
        f"qualified_id={ref.qualified_id}",
        f"role={ref.role.value}",
        f"resolution_mode={ref.resolution_mode.value}",
    )


class SkillPackInspectionProvider:
    """Read-only bound skill pack view for in-flight executions."""

    def __init__(
        self,
        *,
        pinning_store: SkillExecutionPinningStore | None = None,
    ) -> None:
        self._pinning_store = pinning_store

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    def contribute_profile(
        self,
        *,
        resolution: object,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        del resolution, configured_profile_ref
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision(
        self,
        *,
        revision_id: object,
        scope: object,
        revision: object | None,
    ) -> InspectionProviderContribution:
        del revision_id, scope, revision
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        scope_application_id: str,
        scope_tenant_id: str | None,
        binding: object | None,
        pinned_revision: object | None,
    ) -> InspectionProviderContribution:
        del binding, pinned_revision, scope_application_id, scope_tenant_id
        if self._pinning_store is None:
            return InspectionProviderContribution(provider_id=self.provider_id)
        skill_binding = self._pinning_store.get(tenant_id=tenant_id, execution_id=execution_id)
        if skill_binding is None:
            return InspectionProviderContribution(provider_id=self.provider_id)

        pack = skill_binding.resolved_pack
        explanations: list[InspectionExplanation] = []
        extensions: list[InspectionExtensionEvidence] = []

        for ref in pack.resolved_skills:
            explanations.append(
                InspectionExplanation(
                    subject=ref.qualified_id,
                    facts=_ref_facts(ref),
                    reasons=("bound resolved skill",),
                    provenance_refs=(
                        InspectionProvenanceRef(
                            kind="resolved_skill_ref",
                            ref=ref.qualified_id,
                        ),
                    ),
                ),
            )

        tool_lines: dict[str, list[str]] = {}
        prompt_lines: dict[str, list[str]] = {}
        policy_lines: dict[str, list[str]] = {}
        for item in skill_binding.contribution_provenance:
            if item.contribution_kind is SkillContributionKind.TOOL_REQUIREMENT:
                tool_lines.setdefault(item.contribution_id, []).append(item.qualified_id)
            elif item.contribution_kind is SkillContributionKind.PROMPT_INSTRUCTION:
                prompt_lines.setdefault(item.contribution_id, []).append(item.qualified_id)
            elif item.contribution_kind is SkillContributionKind.POLICY_FRAGMENT:
                policy_lines.setdefault(item.contribution_id, []).append(item.qualified_id)

        for tool_id in sorted(tool_lines):
            origins = ",".join(sorted(set(tool_lines[tool_id])))
            extensions.append(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.EXECUTION,
                    subject=f"tool_requirement:{tool_id}",
                    payload={"introduced_by": origins},
                ),
            )
        for prompt_id in sorted(prompt_lines):
            origins = ",".join(sorted(set(prompt_lines[prompt_id])))
            extensions.append(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.EXECUTION,
                    subject=f"prompt_instruction:{prompt_id}",
                    payload={"introduced_by": origins},
                ),
            )
        for fragment_id in sorted(policy_lines):
            origins = ",".join(sorted(set(policy_lines[fragment_id])))
            extensions.append(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.EXECUTION,
                    subject=f"policy_fragment:{fragment_id}",
                    payload={"introduced_by": origins},
                ),
            )

        extensions.append(
            InspectionExtensionEvidence(
                provider_id=self.provider_id,
                scope=InspectionScope.EXECUTION,
                subject="skill_pack",
                payload={
                    "skill_pack_digest": pack.snapshot_digest,
                    "configured_skill_ids": ",".join(skill_binding.configured_skill_ids),
                    "effective_skill_ids": ",".join(pack.skill_ids),
                    "risk_tier": pack.risk_tier.value,
                },
            ),
        )
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=tuple(explanations),
            extension_evidence=tuple(extensions),
        )

    def contribute_capability(
        self,
        *,
        capability_key: str,
        validation: object,
    ) -> InspectionProviderContribution:
        del capability_key, validation
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(
        self,
        *,
        from_revision: object,
        to_revision: object,
    ) -> InspectionProviderContribution:
        del from_revision, to_revision
        return InspectionProviderContribution(provider_id=self.provider_id)


def skill_pack_inspection_provider(
    *,
    pinning_store: SkillExecutionPinningStore | None = None,
) -> RuntimeInspectionProvider:
    return SkillPackInspectionProvider(pinning_store=pinning_store)
