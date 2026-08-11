# © Artur Czarnecki. All rights reserved.

"""WorkspaceAskService — Trusted Ask Workspace orchestration (MVP-2)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.run_metadata import attach_lkw_evidence_metadata
from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.ask_answer_assembler import (
    AskAnswerAssembler,
    index_verified_evidence,
    project_ask_citations,
)
from local_workspace_application.workspaces.ask_models import (
    AskAnswerAssemblyError,
    AskAnswerAssemblyStatus,
    AskError,
    AskRunStatus,
    WorkspaceAskRun,
)
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.knowledge_ask_scope_models import (
    KnowledgeAskScopeError,
    KnowledgeAskScopeV1,
    KnowledgeRetrievalScopeV1,
)
from local_workspace_application.workspaces.knowledge_ask_scope_resolver import (
    KnowledgeAskScopeResolver,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeInspectionService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.search_evidence import (
    SearchEvidenceIncompleteError,
    map_search_hits,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService

logger = logging.getLogger(__name__)


class WorkspaceAskPersistenceError(RuntimeError):
    """Ask run could not be durably persisted."""


class WorkspaceAskNotFoundError(LookupError):
    """Ask run not found for tenant (including cross-tenant)."""


class WorkspaceAskLookupError(LookupError):
    """Workspace authorization failed for Ask (fail-closed HTTP 404)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class WorkspaceAskService:
    """Product orchestration for surface-neutral Ask Workspace."""

    def __init__(
        self,
        *,
        workspace_service: ManagedWorkspaceService,
        workspace_repository: ManagedWorkspaceRepository,
        ask_repository: WorkspaceAskRepository,
        task_executor: LocalWorkspaceTaskExecutor,
        llm_adapter: LLMAdapter | None = None,
        llm_adapter_factory: Callable[[], LLMAdapter] | None = None,
        knowledge_inspection_service: KnowledgeInspectionService | None = None,
        scope_resolver: KnowledgeAskScopeResolver | None = None,
    ) -> None:
        self._workspaces = workspace_service
        self._workspace_repo = workspace_repository
        self._ask_repo = ask_repository
        self._executor = task_executor
        self._llm = llm_adapter
        self._llm_factory = llm_adapter_factory
        self._scope_resolver = scope_resolver
        if self._scope_resolver is None and knowledge_inspection_service is not None:
            self._scope_resolver = KnowledgeAskScopeResolver(knowledge_inspection_service)

    @property
    def llm_adapter(self) -> LLMAdapter:
        if self._llm is None:
            if self._llm_factory is not None:
                self._llm = self._llm_factory()
            else:
                from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter

                self._llm = resolve_llm_adapter(None)
        return self._llm

    @llm_adapter.setter
    def llm_adapter(self, adapter: LLMAdapter) -> None:
        self._llm = adapter

    def use_workspace_authority(
        self,
        workspace_service: ManagedWorkspaceService,
        workspace_repository: ManagedWorkspaceRepository | None = None,
    ) -> None:
        """Bind Ask to the same managed-workspace authority used by listing/GET."""
        self._workspaces = workspace_service
        if workspace_repository is None:
            workspace_repository = workspace_service.repository
        if self._workspace_repo is not workspace_repository:
            logger.warning(
                "ask_workspace_authority_realigned reason=repository_inconsistency"
            )
            self._workspace_repo = workspace_repository
            if self._ask_repo.document_store is not workspace_repository.document_store:
                self._ask_repo = WorkspaceAskRepository(workspace_repository.document_store)

    async def ask(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        question: str,
        limit: int = 10,
        knowledge_scope: KnowledgeAskScopeV1 | None = None,
    ) -> WorkspaceAskRun:
        workspace = self._workspaces.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            reason = self._classify_workspace_lookup_failure(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            logger.warning("ask_workspace_lookup_failed reason=%s", reason)
            raise WorkspaceAskLookupError(reason)

        run_id = new_run_id()
        created_at = datetime.now(UTC)
        run = WorkspaceAskRun(
            run_id=run_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            question=question,
            knowledge_item_ids=(
                knowledge_scope.knowledge_item_ids if knowledge_scope is not None else None
            ),
            status=AskRunStatus.FAILED,
            evidence=[],
            answer=None,
            citations=[],
            created_at=created_at,
            completed_at=None,
            error=None,
        )
        self._persist(run)

        retrieval_scope: KnowledgeRetrievalScopeV1 | None = None
        if knowledge_scope is not None:
            if self._scope_resolver is None:
                return self._finalize_failed(
                    run,
                    code="knowledge_ask_scope_invalid",
                    message="scoped ask is unavailable",
                )
            try:
                retrieval_scope = self._scope_resolver.resolve(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    scope=knowledge_scope,
                )
            except KnowledgeAskScopeError as exc:
                return self._finalize_failed(
                    run,
                    code=exc.error_code,
                    message=exc.message,
                )

        try:
            evidence = await self._retrieve_verified_evidence(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                question=question,
                limit=limit,
                allowed_source_ids=(
                    retrieval_scope.allowed_source_ids if retrieval_scope is not None else None
                ),
            )
        except SearchEvidenceIncompleteError:
            self._finalize_failed(
                run,
                code="search_evidence_incomplete",
                message="search evidence could not be verified",
            )
            raise
        except Exception as exc:
            self._finalize_failed(
                run,
                code="search_failed",
                message=f"search failed: {exc.__class__.__name__}",
                cause=exc,
            )
            raise

        if retrieval_scope is not None:
            try:
                self._validate_scoped_evidence(
                    evidence,
                    allowed_source_ids=retrieval_scope.allowed_source_ids,
                )
            except KnowledgeAskScopeError as exc:
                return self._finalize_failed(
                    run,
                    code=exc.error_code,
                    message=exc.message,
                )

        run = run.model_copy(update={"evidence": evidence})

        if not evidence:
            return self._finalize(
                run,
                status=AskRunStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                citations=[],
                error=None,
            )

        assembler = AskAnswerAssembler(self.llm_adapter)
        try:
            assembly = assembler.assemble(question=question, evidence=evidence)
        except AskAnswerAssemblyError as exc:
            return self._finalize_failed(
                run,
                code=exc.code,
                message=exc.message,
                cause=exc,
            )
        except Exception as exc:
            return self._finalize_failed(
                run,
                code="assembly_failed",
                message=f"answer assembly failed: {exc.__class__.__name__}",
                cause=exc,
            )

        if assembly.status == AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE:
            return self._finalize(
                run,
                status=AskRunStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                citations=[],
                error=None,
            )

        indexed = index_verified_evidence(evidence)
        try:
            citations = project_ask_citations(
                used_evidence_ids=assembly.used_evidence_ids,
                indexed_evidence=indexed,
            )
        except AskAnswerAssemblyError as exc:
            return self._finalize_failed(
                run,
                code=exc.code,
                message=exc.message,
                cause=exc,
            )

        if not citations or not (assembly.answer or "").strip():
            return self._finalize_failed(
                run,
                code="completed_without_citation",
                message="completed answer requires at least one verified citation",
            )

        return self._finalize(
            run,
            status=AskRunStatus.COMPLETED,
            answer=assembly.answer,
            citations=citations,
            error=None,
        )

    def get_run(self, *, tenant_id: str, run_id: str) -> WorkspaceAskRun:
        run = self._ask_repo.get_run(tenant_id=tenant_id, run_id=run_id)
        if run is None:
            raise WorkspaceAskNotFoundError(run_id)
        return run

    async def _retrieve_verified_evidence(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        question: str,
        limit: int,
        allowed_source_ids: tuple[str, ...] | None = None,
    ) -> list[WorkspaceSearchHitV1]:
        metadata: dict[str, Any] = {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "collection_id": workspace_id,
            "query": question,
            "top_k": max(limit, 10),
            "requested_by": "lkw.managed_workspace.ask",
        }
        if allowed_source_ids:
            metadata["allowed_source_ids"] = list(allowed_source_ids)
        task = Task(
            task_id=new_run_id(),
            tenant_id=tenant_id,
            user_id="lkw.managed_workspace",
            message=question,
            context=TaskContext(capability="local.workspace.search"),
            metadata=metadata,
        )
        result = await self._executor.execute(task)
        result_metadata = dict(getattr(result, "metadata", None) or {})
        attach_lkw_evidence_metadata(
            result_metadata,
            task_result=result,
            capability="local.workspace.search",
        )
        result = result.model_copy(update={"metadata": result_metadata})
        return map_search_hits(
            repository=self._workspace_repo,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            task_result=result,
            limit=limit,
        )

    def _finalize(
        self,
        run: WorkspaceAskRun,
        *,
        status: AskRunStatus,
        answer: str | None,
        citations: list[Any],
        error: AskError | None,
    ) -> WorkspaceAskRun:
        finalized = run.model_copy(
            update={
                "status": status,
                "answer": answer,
                "citations": citations,
                "error": error,
                "completed_at": datetime.now(UTC),
            }
        )
        self._persist(finalized)
        return finalized

    def _finalize_failed(
        self,
        run: WorkspaceAskRun,
        *,
        code: str,
        message: str,
        cause: BaseException | None = None,
    ) -> WorkspaceAskRun:
        _ = cause
        return self._finalize(
            run,
            status=AskRunStatus.FAILED,
            answer=None,
            citations=[],
            error=AskError(code=code, message=message),
        )

    def _persist(self, run: WorkspaceAskRun) -> None:
        try:
            self._ask_repo.put_run(run)
        except Exception as exc:
            raise WorkspaceAskPersistenceError(
                f"ask run persistence failed: {exc.__class__.__name__}"
            ) from exc

    def _classify_workspace_lookup_failure(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> str:
        """Return a safe diagnostic code without tenant/workspace identifiers."""
        wanted = (workspace_id or "").strip()
        if self._workspace_repo is not self._workspaces.repository:
            return "repository_inconsistency"
        if (
            self._ask_repo.document_store
            is not self._workspace_repo.document_store
        ):
            return "repository_inconsistency"
        try:
            listed = self._workspaces.list_workspaces(tenant_id=tenant_id)
        except Exception:  # noqa: BLE001 — classification must not raise
            return "workspace_lookup_failed"
        if any((item.workspace_id or "").strip() == wanted for item in listed):
            return "repository_inconsistency"
        return "workspace_lookup_failed"

    @staticmethod
    def _validate_scoped_evidence(
        evidence: list[WorkspaceSearchHitV1],
        *,
        allowed_source_ids: tuple[str, ...],
    ) -> None:
        allowed = frozenset(allowed_source_ids)
        for item in evidence:
            source_id = str(item.source_id or "").strip()
            if source_id not in allowed:
                raise KnowledgeAskScopeError(
                    "knowledge_ask_scope_integrity_failed",
                    "scoped evidence integrity check failed",
                )
