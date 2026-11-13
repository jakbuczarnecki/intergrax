# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""The ONE governed access port in front of every artifact store (ADR-PDC-08).

Authorization is delegated to the existing runtime governance owner
(``AgentRuntimeGovernancePort``: capability grant -> policy -> decision, audited by its pipeline).
The tenant check and the per-operation audit event (``GovernanceAuditEvent`` through the existing
``GovernanceAuditRecorder``) are done here. Stores stay storage-only and never see a security
decision. This module adds no policy engine, no registry and no authority of its own.

Governed secure download (OD-PDC-SCOPE-04 = A) is this port plus a short-lived provider URL;
``presigned_url`` is called nowhere else in product code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    PolicyEvaluationResult,
    ToolAuthorizationDecision,
    ToolAuthorizationDecisionState,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestMismatchError,
    DurableArtifactRef,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder
from intergrax.runtime.agent_governance.errors import (
    CapabilityNotGrantedError,
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.nexus.artifacts.store_base import ArtifactDeletionRefused
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity

CAPABILITY_READ = "artifact.read"
CAPABILITY_DOWNLOAD = "artifact.download"
CAPABILITY_LEGAL_HOLD = "artifact.legal_hold"
CAPABILITY_DELETE = "artifact.delete"
ARTIFACT_CAPABILITIES: tuple[str, ...] = (
    CAPABILITY_READ,
    CAPABILITY_DOWNLOAD,
    CAPABILITY_LEGAL_HOLD,
    CAPABILITY_DELETE,
)
_OUTCOME_SUFFIX = ".outcome"
DEFAULT_MAX_DOWNLOAD_TTL_SECONDS = 300

_RISK: Mapping[str, ToolAuthorizationRiskLevel] = {
    CAPABILITY_READ: ToolAuthorizationRiskLevel.LOW,
    CAPABILITY_DOWNLOAD: ToolAuthorizationRiskLevel.MEDIUM,
    CAPABILITY_LEGAL_HOLD: ToolAuthorizationRiskLevel.HIGH,
    CAPABILITY_DELETE: ToolAuthorizationRiskLevel.HIGH,
}


# ---------------------------------------------------------------------------- errors


class ArtifactAccessError(Exception):
    """Base class of governed artifact access failures."""


class ArtifactAccessDenied(ArtifactAccessError, PermissionError):
    """Access was denied; ``reason_code`` is one of the stable codes below."""

    def __init__(self, reason_code: str, artifact_id: str, detail: str = "") -> None:
        super().__init__(f"artifact access denied ({reason_code}): {artifact_id}")
        self.reason_code = reason_code
        self.artifact_id = artifact_id
        self.detail = detail


class ArtifactNotFound(ArtifactAccessError, LookupError):
    """The artifact does not exist in the caller's tenant (and class)."""


class ArtifactDownloadUnavailable(ArtifactAccessError):
    """No download path is configured for this artifact."""


# ---------------------------------------------------------------------------- value types


@dataclass(frozen=True)
class ArtifactAccessContext:
    """Who is accessing, and in which execution (the evidence anchor of every operation)."""

    agent: AgentIdentity
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId | None = None

    @property
    def tenant_id(self) -> str:
        return self.agent.tenant_id

    @classmethod
    def for_background_execution(cls, *, agent_id: str, identity: BackgroundExecutionIdentity) -> ArtifactAccessContext:
        """Context of a system actor running as admitted background work."""
        return cls(
            agent=AgentIdentity(agent_id=agent_id, tenant_id=identity.tenant_id),
            task_id=identity.task_id,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            execution_id=identity.execution_id,
        )


@dataclass(frozen=True)
class ArtifactReadRequest:
    artifact_class: ArtifactClass
    tenant_id: str
    artifact_id: str
    version: int | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class GovernedArtifactContent:
    ref: DurableArtifactRef
    content: bytes


@dataclass(frozen=True)
class GovernedDownloadGrant:
    url: str
    expires_at: datetime
    expires_in_seconds: int
    tenant_id: str
    artifact_id: str


# ---------------------------------------------------------------------------- source ports


@runtime_checkable
class ArtifactReadSource(Protocol):
    """Storage-only read view of one store. Contains no security logic."""

    def read(self, request: ArtifactReadRequest) -> GovernedArtifactContent | None: ...


@runtime_checkable
class ArtifactDownloadSource(Protocol):
    """Optional: a source whose content lives in object storage exposes its object key."""

    def download_key(self, request: ArtifactReadRequest) -> str | None: ...


@runtime_checkable
class ArtifactLifecycleSource(Protocol):
    """Optional: a source whose store supports legal hold and deletion."""

    def place_legal_hold(self, request: ArtifactReadRequest, *, placed_by: str, reason: str) -> None: ...

    def release_legal_hold(self, request: ArtifactReadRequest) -> None: ...

    def delete(self, request: ArtifactReadRequest) -> None: ...


# ---------------------------------------------------------------------------- evidence


class ArtifactOperationAuditor:
    """Records one ``GovernanceAuditEvent`` per artifact operation outcome."""

    def __init__(self, recorder: GovernanceAuditRecorder) -> None:
        self._recorder = recorder

    def record(
        self,
        context: ArtifactAccessContext,
        *,
        capability: str,
        artifact_id: str,
        decision: ToolAuthorizationDecisionState,
        check: str,
        detail: str,
        at: datetime | None = None,
    ) -> None:
        request = _authorization_request(
            context,
            capability=capability + _OUTCOME_SUFFIX,
            artifact_id=artifact_id,
            requested_action=capability,
            risk=_RISK.get(capability, ToolAuthorizationRiskLevel.LOW),
        )
        self._recorder.record_decision(
            request,
            ToolAuthorizationDecision(
                decision=decision,
                reason=f"{check}:{artifact_id}",
                policy_results=(
                    PolicyEvaluationResult(
                        policy_id=f"artifact.port.{check}",
                        decision=decision,
                        reason=detail,
                    ),
                ),
            ),
            timestamp=at,
        )


def _authorization_request(
    context: ArtifactAccessContext,
    *,
    capability: str,
    artifact_id: str,
    requested_action: str,
    risk: ToolAuthorizationRiskLevel,
) -> ToolAuthorizationRequest:
    return ToolAuthorizationRequest(
        agent=context.agent,
        task_id=context.task_id,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        execution_id=context.execution_id,
        capability=capability,
        tool_id=f"artifact:{artifact_id}",
        requested_action=requested_action,
        risk_level=risk,
    )


# ---------------------------------------------------------------------------- the port


class GovernedArtifactReadPort:
    """Authorization + tenant check + audit in front of every artifact source."""

    def __init__(
        self,
        *,
        governance: AgentRuntimeGovernancePort,
        audit: GovernanceAuditRecorder,
        sources: Mapping[ArtifactClass, ArtifactReadSource],
        download_issuer: ObjectStorage | None = None,
        max_download_ttl_seconds: int = DEFAULT_MAX_DOWNLOAD_TTL_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_download_ttl_seconds < 1:
            raise ValueError("max_download_ttl_seconds must be positive")
        self._governance = governance
        self._auditor = ArtifactOperationAuditor(audit)
        self._sources = dict(sources)
        self._download_issuer = download_issuer
        self._max_download_ttl = max_download_ttl_seconds
        self._clock = clock or SystemTimeProvider.utc_now

    # ---- operations -------------------------------------------------------

    def read(self, context: ArtifactAccessContext, request: ArtifactReadRequest) -> GovernedArtifactContent:
        def action(source: ArtifactReadSource) -> tuple[GovernedArtifactContent, str]:
            found = source.read(request)
            if found is None:
                raise ArtifactNotFound(request.artifact_id)
            if found.ref.scope.tenant_id != context.tenant_id:
                raise ArtifactAccessDenied("tenant_mismatch", request.artifact_id, "stored tenant differs")
            if request.workspace_id is not None and found.ref.scope.workspace_id != request.workspace_id:
                raise ArtifactAccessDenied("workspace_mismatch", request.artifact_id, "stored workspace differs")
            found.ref.verify_content(found.content)
            return found, (
                f"class={request.artifact_class.value};version={found.ref.version};digest={found.ref.content_digest}"
            )

        return self._guard(context, request, CAPABILITY_READ, action)

    def issue_download_url(
        self,
        context: ArtifactAccessContext,
        request: ArtifactReadRequest,
        *,
        ttl_seconds: int,
    ) -> GovernedDownloadGrant:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        ttl = min(ttl_seconds, self._max_download_ttl)

        def action(source: ArtifactReadSource) -> tuple[GovernedDownloadGrant, str]:
            if self._download_issuer is None or not isinstance(source, ArtifactDownloadSource):
                raise ArtifactDownloadUnavailable(request.artifact_id)
            key = source.download_key(request)
            if key is None:
                raise ArtifactNotFound(request.artifact_id)
            url = self._download_issuer.presigned_url(key, expires_in_seconds=ttl, method="GET")
            grant = GovernedDownloadGrant(
                url=url,
                expires_at=self._clock() + timedelta(seconds=ttl),
                expires_in_seconds=ttl,
                tenant_id=context.tenant_id,
                artifact_id=request.artifact_id,
            )
            return grant, f"class={request.artifact_class.value};ttl_seconds={ttl}"

        return self._guard(context, request, CAPABILITY_DOWNLOAD, action)

    def place_legal_hold(self, context: ArtifactAccessContext, request: ArtifactReadRequest, *, reason: str) -> None:
        def action(source: ArtifactReadSource) -> tuple[None, str]:
            lifecycle = _lifecycle(source, request)
            lifecycle.place_legal_hold(request, placed_by=context.agent.agent_id, reason=reason)
            return None, f"legal_hold_placed;reason={reason}"

        self._guard(context, request, CAPABILITY_LEGAL_HOLD, action)

    def release_legal_hold(self, context: ArtifactAccessContext, request: ArtifactReadRequest) -> None:
        def action(source: ArtifactReadSource) -> tuple[None, str]:
            _lifecycle(source, request).release_legal_hold(request)
            return None, "legal_hold_released"

        self._guard(context, request, CAPABILITY_LEGAL_HOLD, action)

    def delete_artifact(self, context: ArtifactAccessContext, request: ArtifactReadRequest) -> None:
        def action(source: ArtifactReadSource) -> tuple[None, str]:
            _lifecycle(source, request).delete(request)
            return None, "deleted"

        self._guard(context, request, CAPABILITY_DELETE, action)

    # ---- the single gate ----------------------------------------------------

    def _guard(self, context, request, capability, action):
        # 1. tenant check, before anything else touches a store
        if request.tenant_id != context.tenant_id:
            self._deny(context, request, capability, "tenant_mismatch", "request tenant differs from caller tenant")
            raise ArtifactAccessDenied("tenant_mismatch", request.artifact_id)
        # 2. fail closed when no store is registered for the class
        source = self._sources.get(request.artifact_class)
        if source is None:
            self._deny(context, request, capability, "no_source", f"no store for {request.artifact_class.value}")
            raise ArtifactAccessDenied("no_source", request.artifact_id)
        # 3. authorization through the existing governance owner
        try:
            self._governance.authorize_tool(
                _authorization_request(
                    context,
                    capability=capability,
                    artifact_id=request.artifact_id,
                    requested_action=f"{capability}:{request.artifact_class.value}",
                    risk=_RISK[capability],
                )
            )
        except (CapabilityNotGrantedError, ToolGovernanceDeniedError) as exc:
            self._deny(context, request, capability, "not_authorized", type(exc).__name__)
            raise ArtifactAccessDenied("not_authorized", request.artifact_id) from exc
        except ToolGovernanceApprovalRequiredError as exc:
            self._deny(context, request, capability, "approval_required", type(exc).__name__)
            raise ArtifactAccessDenied("approval_required", request.artifact_id) from exc
        # 4. the operation itself, with its outcome in evidence
        try:
            result, detail = action(source)
        except ArtifactAccessDenied as exc:
            self._deny(context, request, capability, exc.reason_code, exc.detail or exc.reason_code)
            raise
        except ArtifactNotFound:
            self._deny(context, request, capability, "not_found", "no such artifact in the caller tenant")
            raise
        except KeyError as exc:
            self._deny(context, request, capability, "not_found", "no such artifact in the caller tenant")
            raise ArtifactNotFound(request.artifact_id) from exc
        except ArtifactDeletionRefused as exc:
            self._deny(context, request, capability, "deletion_refused", f"{exc.reason};artifact_id={exc.artifact_id}")
            raise
        except ArtifactDigestMismatchError:
            self._deny(context, request, capability, "integrity_violation", "content does not match its digest")
            raise
        except ArtifactDownloadUnavailable:
            self._deny(context, request, capability, "download_unavailable", "no object-storage path")
            raise
        self._auditor.record(
            context,
            capability=capability,
            artifact_id=request.artifact_id,
            decision=ToolAuthorizationDecisionState.ALLOW,
            check="completed",
            detail=detail,
            at=self._clock(),
        )
        return result

    def _deny(self, context, request, capability, check, detail) -> None:
        self._auditor.record(
            context,
            capability=capability,
            artifact_id=request.artifact_id,
            decision=ToolAuthorizationDecisionState.DENY,
            check=check,
            detail=detail,
            at=self._clock(),
        )


def _lifecycle(source: ArtifactReadSource, request: ArtifactReadRequest) -> ArtifactLifecycleSource:
    if not isinstance(source, ArtifactLifecycleSource):
        raise ArtifactAccessDenied("operation_unsupported", request.artifact_id)
    return source
