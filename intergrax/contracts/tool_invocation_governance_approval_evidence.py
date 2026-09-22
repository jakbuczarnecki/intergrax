# © Artur Czarnecki. All rights reserved.

"""Provider-neutral governance approval evidence for catalog tool invocation."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.autonomous_work._validation import require_non_empty_text


@dataclass(frozen=True, slots=True)
class ToolInvocationGovernanceApprovalEvidence:
    """Scoped human/policy approval for one execution-bound catalog tool invocation.

    Carries the minimum identity required for ``ToolAuthorizationRequest.approval_evidence_ref``
    and downstream scope correlation. Does not execute or validate policy.
    """

    evidence_ref: str
    invocation_scope_id: str
    tenant_id: str
    task_id: str
    run_id: str
    step_id: str
    tool_id: str
    agent_id: str
    idempotency_key: str | None = None
    matched_rule_ids: tuple[str, ...] = ()
    human_request_id: str = ""
    policy_provenance_digest: str | None = None
    pause_id: str = ""
    approved_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_ref",
            require_non_empty_text(self.evidence_ref, label="evidence_ref"),
        )
        object.__setattr__(
            self,
            "invocation_scope_id",
            require_non_empty_text(
                self.invocation_scope_id,
                label="invocation_scope_id",
            ),
        )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        for label, value in (
            ("task_id", self.task_id),
            ("run_id", self.run_id),
            ("step_id", self.step_id),
            ("tool_id", self.tool_id),
            ("agent_id", self.agent_id),
        ):
            object.__setattr__(
                self,
                label,
                require_non_empty_text(value, label=label),
            )


__all__ = [
    "ToolInvocationGovernanceApprovalEvidence",
]
