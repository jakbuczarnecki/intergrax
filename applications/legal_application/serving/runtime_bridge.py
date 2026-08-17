# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Maps Legal API v1 ↔ :class:`RuntimeRequest` / :class:`RuntimeAnswer`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from legal_application.serving.schemas import (
    AttachmentRefV1,
    LegalChatRequestV1,
    LegalChatResponseV1,
)
from intergrax.fastapi_core.context import RequestContext
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.llm.messages import AttachmentRef
from intergrax.runtime.nexus.policies.runtime_policies import ApiTraceExportMode, DataCompliancePolicy
from intergrax.runtime.nexus.responses.response_schema import (
    HistoryCompressionStrategy,
    RuntimeAnswer,
    RuntimeRequest,
)


class LegalApiV1RuntimeMapper:
    """
    Converts between product HTTP v1 payloads and Nexus runtime dataclasses.

    Stateless aside from class-level contract constants; safe to reuse as a singleton.
    """

    API_METADATA_KEY = "api"
    API_PRODUCT = "legal_agent"
    API_VERSION = "1"

    def to_runtime_request(
        self,
        body: LegalChatRequestV1,
        *,
        http_context: RequestContext,
        default_agent_id: str,
        tenant_id: str,
        user_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> RuntimeRequest:
        agent_id = (body.agent_id or default_agent_id).strip()
        meta = dict(body.metadata)
        meta[self.API_METADATA_KEY] = {"product": self.API_PRODUCT, "version": self.API_VERSION}
        meta["http_request_id"] = http_context.request_id

        attachments = [self._attachment_from_v1(a) for a in body.attachments]

        return RuntimeRequest(
            agent_id=agent_id,
            user_id=user_id,
            session_id=body.session_id.strip(),
            message=body.message,
            task_id=task_id,
            run_id=run_id,
            attachments=attachments,
            workspace_id=body.workspace_id,
            tenant_id=tenant_id,
            metadata=meta,
            instructions=body.instructions,
            history_compression_strategy=self._history_compression(body.history_compression),
            max_output_tokens=body.max_output_tokens,
        )

    def to_legal_chat_response(
        self,
        answer: RuntimeAnswer,
        *,
        http_context: RequestContext,
        include_trace: bool,
        data_compliance: Optional[DataCompliancePolicy] = None,
    ) -> LegalChatResponseV1:
        policy = data_compliance or DataCompliancePolicy()
        trace_payload = self._trace_events_for_api(
            answer.trace_events,
            include_trace_requested=include_trace,
            mode=policy.api_trace_export,
        )
        redact_tc = policy.redact_tool_calls_in_api

        llm_usage: Optional[Dict[str, Any]] = None
        if answer.llm_usage_report is not None:
            llm_usage = answer.llm_usage_report.to_dict()

        return LegalChatResponseV1(
            request_id=http_context.request_id,
            run_id=answer.run_id,
            stop_reason=answer.stop_reason.value,
            answer=answer.answer,
            route=self._route_to_dict(answer.route),
            stats=self._stats_to_dict(answer.stats),
            citations=[self._citation_to_dict(c) for c in (answer.citations or [])],
            tool_calls=[
                self._tool_call_to_dict(t, redact_arguments=redact_tc) for t in (answer.tool_calls or [])
            ],
            llm_usage=llm_usage,
            trace_events=trace_payload,
        )

    @staticmethod
    def _trace_events_for_api(
        events: Optional[Sequence[Any]],
        *,
        include_trace_requested: bool,
        mode: ApiTraceExportMode,
    ) -> Optional[List[Dict[str, Any]]]:
        if not events or mode == "none" or not include_trace_requested:
            return None
        if mode == "full":
            return [e.to_dict() for e in events]
        return [e.with_redacted_payload().to_dict() for e in events]

    @staticmethod
    def _attachment_from_v1(ref: AttachmentRefV1) -> AttachmentRef:
        return AttachmentRef(
            id=ref.id,
            type=ref.type,
            uri=ref.uri,
            metadata=dict(ref.metadata),
        )

    @staticmethod
    def _history_compression(name: Optional[str]) -> HistoryCompressionStrategy:
        if not name:
            return HistoryCompressionStrategy.TRUNCATE_OLDEST
        try:
            return HistoryCompressionStrategy[name.upper()]
        except KeyError:
            return HistoryCompressionStrategy.TRUNCATE_OLDEST

    @staticmethod
    def _citation_to_dict(c: Any) -> Dict[str, Any]:
        return {
            "source_id": c.source_id,
            "source_type": c.source_type,
            "source_label": c.source_label,
            "url": c.url,
            "score": c.score,
            "extra": dict(c.extra) if c.extra else {},
        }

    @staticmethod
    def _tool_call_to_dict(t: Any, *, redact_arguments: bool = False) -> Dict[str, Any]:
        args: Dict[str, Any] = dict(t.arguments) if t.arguments else {}
        if redact_arguments:
            args = {"_redacted": True}
        return {
            "tool_name": t.tool_name,
            "arguments": args,
            "result_summary": t.result_summary,
            "success": t.success,
            "error_message": t.error_message,
            "extra": dict(t.extra) if t.extra else {},
        }

    @staticmethod
    def _route_to_dict(route: Any) -> Dict[str, Any]:
        return {
            "used_rag": route.used_rag,
            "used_websearch": route.used_websearch,
            "used_tools": route.used_tools,
            "used_user_profile": route.used_user_profile,
            "used_user_longterm_memory": route.used_user_longterm_memory,
            "strategy": route.strategy,
            "extra": dict(route.extra) if route.extra else {},
        }

    @staticmethod
    def _stats_to_dict(stats: Any) -> Dict[str, Any]:
        return {
            "total_tokens": stats.total_tokens,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "rag_tokens": stats.rag_tokens,
            "websearch_tokens": stats.websearch_tokens,
            "tool_tokens": stats.tool_tokens,
            "duration_ms": stats.duration_ms,
            "extra": dict(stats.extra) if stats.extra else {},
        }
