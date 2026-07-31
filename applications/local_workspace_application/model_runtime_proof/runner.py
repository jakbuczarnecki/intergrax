# © Artur Czarnecki. All rights reserved.

"""Orchestration for LKW Ollama / vLLM model runtime portability proof."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.catalog_capabilities import (
    unwrap_catalog_capability_adapter,
)

from local_workspace_application.model_runtime_proof.aggregation import (
    index_invariance_passes,
    provider_qualification_passes,
)
from local_workspace_application.model_runtime_proof.config import (
    ModelRuntimeProofConfig,
    load_proof_config_from_env,
)
from local_workspace_application.model_runtime_proof.contracts import (
    FIXTURE_MARKER,
    FixtureRecord,
    IndexInvarianceResult,
    ModelRuntimeProofResult,
    ProofFailureCode,
    ProofOverallStatus,
    ProviderQualificationResult,
    StageStatus,
)
from local_workspace_application.model_runtime_proof.fixture import (
    IndexedFixture,
    index_managed_file_fixture_async,
)
from local_workspace_application.model_runtime_proof.health import probe_provider_health
from local_workspace_application.model_runtime_proof.index_identity import (
    capture_index_identity,
    compare_embedding_identity,
    compare_index_identity,
    index_identity_is_complete,
)
from local_workspace_application.model_runtime_proof.report import (
    render_terminal_summary,
    stale_evidence_notice,
    write_evidence,
)
from local_workspace_application.model_runtime_proof.repository_state import (
    capture_repository_state,
)
from local_workspace_application.model_runtime_proof.runtime import (
    build_proof_runtime_session,
)
from local_workspace_application.model_runtime_proof.stages import (
    count_indexing_operations,
    run_basic_generation,
    run_grounded_ask,
    run_structured_planning,
    run_tool_call_and_execution,
)

_REJECTED_OLLAMA_MODEL = "qwen2.5:7b"


class ModelRuntimeProofRunner:
    def __init__(self, config: ModelRuntimeProofConfig) -> None:
        self._config = config
        self._document_store = InMemoryDocumentStore()
        self._data_home: Path | None = None
        self._indexing_count_before = 0

    async def run(self) -> ModelRuntimeProofResult:
        started = datetime.now(UTC)
        proof_id = f"lkw-model-runtime-proof:{secrets.token_hex(6)}"
        repository_state = capture_repository_state()
        result = ModelRuntimeProofResult(
            proof_id=proof_id,
            started_at=started,
            repository_commit=repository_state.repository_head_at_proof,
            repository_state=repository_state,
            vllm_provisioning_classification=self._config.vllm_provisioning_classification,
        )

        validation_errors = self._config.validate()
        if validation_errors:
            return result.model_copy(
                update={
                    "completed_at": datetime.now(UTC),
                    "overall_status": ProofOverallStatus.BLOCKED,
                    "limitations": ("invalid_proof_configuration",),
                }
            )

        if self._config.data_home:
            self._data_home = Path(self._config.data_home)
        else:
            self._data_home = Path.cwd() / "build" / "lkw-model-runtime-proof"

        index_session = build_proof_runtime_session(
            self._config,
            provider=None,
            document_store=self._document_store,
            data_home=self._data_home,
        )
        try:
            fixture = await index_managed_file_fixture_async(
                index_session,
                tenant_id=self._config.tenant_id,
            )
            index_identity = capture_index_identity(
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
                repository=index_session.repository,
                wiring_context=index_session.wiring_context,
                embedding_manager=index_session.embedding_manager,
            )
            if not index_identity_is_complete(index_identity):
                return result.model_copy(
                    update={
                        "completed_at": datetime.now(UTC),
                        "fixture": self._fixture_record(fixture),
                        "index_identity": index_identity,
                        "overall_status": ProofOverallStatus.FAIL,
                        "limitations": ("index_identity_incomplete",),
                    }
                )
            self._indexing_count_before = count_indexing_operations(
                index_session.repository,
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
            )
        except Exception as exc:
            return result.model_copy(
                update={
                    "completed_at": datetime.now(UTC),
                    "overall_status": ProofOverallStatus.FAIL,
                    "limitations": (
                        f"index_fixture_failed:{exc.__class__.__name__}:{exc}",
                    ),
                }
            )
        finally:
            index_session.close()

        embedding_before = index_identity.embedding
        provider_results: dict[str, ProviderQualificationResult] = {}
        overall_pass = True
        adapter_ids: dict[str, str] = {}

        for provider in ("ollama", "vllm"):
            provider_result = await self._qualify_provider(
                provider=provider,  # type: ignore[arg-type]
                fixture=fixture,
                index_identity=index_identity,
            )
            provider_results[provider] = provider_result
            if provider_result.session_adapter_object_id is not None:
                adapter_ids[provider] = provider_result.session_adapter_object_id
            if not provider_qualification_passes(provider_result):
                overall_pass = False

        if len(adapter_ids) == 2 and adapter_ids["ollama"] == adapter_ids["vllm"]:
            overall_pass = False

        after_session = build_proof_runtime_session(
            self._config,
            provider="vllm",
            document_store=self._document_store,
            data_home=self._data_home,
        )
        try:
            after_identity = capture_index_identity(
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
                repository=after_session.repository,
                wiring_context=after_session.wiring_context,
                embedding_manager=after_session.embedding_manager,
            )
            indexing_after = count_indexing_operations(
                after_session.repository,
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
            )
        finally:
            after_session.close()

        comparison = compare_index_identity(index_identity, after_identity)
        embedding_ok = compare_embedding_identity(
            embedding_before, after_identity.embedding
        )
        no_reindex_ok = indexing_after == self._indexing_count_before

        index_invariance = IndexInvarianceResult(
            embedding_identity=StageStatus.PASS if embedding_ok else StageStatus.FAIL,
            collection_identity=StageStatus.PASS
            if comparison.collection_identity
            else StageStatus.FAIL,
            vector_count=StageStatus.PASS
            if comparison.vector_count
            else StageStatus.FAIL,
            source_identity=StageStatus.PASS
            if comparison.source_id
            else StageStatus.FAIL,
            document_identity=StageStatus.PASS
            if comparison.document_id
            else StageStatus.FAIL,
            content_hash=StageStatus.PASS
            if comparison.content_hash
            else StageStatus.FAIL,
            chunk_count=StageStatus.PASS
            if comparison.chunk_count
            else StageStatus.FAIL,
            no_reindex=StageStatus.PASS if no_reindex_ok else StageStatus.FAIL,
        )
        if not index_invariance_passes(index_invariance):
            overall_pass = False

        if self._config.vllm_provisioning_classification == "unverified":
            overall_pass = False

        limitations: list[str] = [
            "exact configured Ollama and vLLM pairs only; not universal model parity",
            "runtime hot swapping not required or proven",
        ]
        if self._config.vllm_provisioning_classification == "unverified":
            limitations.append("vllm_provisioning_classification_unverified")
        if self._config.ollama_model != _REJECTED_OLLAMA_MODEL:
            limitations.append(
                f"{_REJECTED_OLLAMA_MODEL} was not the qualified full-product Ollama model"
            )

        completed = datetime.now(UTC)
        return result.model_copy(
            update={
                "completed_at": completed,
                "fixture": self._fixture_record(fixture),
                "index_identity": index_identity,
                "embedding_identity_before": embedding_before,
                "embedding_identity_after": after_identity.embedding,
                "provider_results": provider_results,
                "index_invariance": index_invariance,
                "overall_status": ProofOverallStatus.PASS
                if overall_pass
                else ProofOverallStatus.FAIL,
                "limitations": tuple(limitations),
            }
        )

    def _fixture_record(self, fixture: IndexedFixture) -> FixtureRecord:
        return FixtureRecord(
            marker=FIXTURE_MARKER,
            tenant_id=fixture.tenant_id,
            workspace_id=fixture.workspace_id,
            input_id=fixture.input_id,
            source_id=fixture.source_id,
            operation_id=fixture.operation_id,
            document_id=fixture.document_id,
            content_hash=fixture.content_hash,
            indexing_operations=self._indexing_count_before,
            chunk_count=fixture.chunk_count,
        )

    def _assert_adapter_identity(
        self,
        adapter: LLMAdapter,
        *,
        provider: Literal["ollama", "vllm"],
        resolved_model: str | None,
        session: object,
    ) -> tuple[bool, str | None]:
        session_adapter = getattr(session, "llm_adapter", None)
        if session_adapter is None or adapter is not session_adapter:
            return False, "session_adapter_mismatch"

        ask_service = getattr(session, "ask_service", None)
        if ask_service is None:
            return False, "ask_service_missing"

        core = unwrap_catalog_capability_adapter(adapter)
        ask_core = unwrap_catalog_capability_adapter(ask_service.llm_adapter)
        if id(core) != id(ask_core):
            return False, "ask_adapter_mismatch"

        provider_name = core._provider_slug()
        if provider_name != provider:
            return False, "provider_mismatch"

        model_name = str(getattr(core, "model", "") or "")
        expected_model = resolved_model or ""
        if expected_model and model_name and model_name != expected_model:
            return False, "model_mismatch"
        return True, None

    async def _qualify_provider(
        self,
        *,
        provider: Literal["ollama", "vllm"],
        fixture: IndexedFixture,
        index_identity,
    ) -> ProviderQualificationResult:
        configured_model = (
            self._config.ollama_model
            if provider == "ollama"
            else self._config.vllm_model
        )
        base = ProviderQualificationResult(
            provider=provider,
            configured_model=configured_model,
        )
        health, failure, detail = probe_provider_health(provider, self._config)
        if health is None:
            return base.model_copy(
                update={
                    "health_status": StageStatus.FAIL,
                    "failure_code": failure,
                    "safe_error_excerpt": detail,
                }
            )

        session = build_proof_runtime_session(
            self._config,
            provider=provider,
            document_store=self._document_store,
            data_home=self._data_home,
        )
        try:
            adapter = session.llm_adapter
            if adapter is None:
                return base.model_copy(
                    update={
                        "health_status": StageStatus.PASS,
                        "resolved_through_canonical_resolver": False,
                        "failure_code": failure,
                        "safe_error_excerpt": "missing_canonical_adapter",
                    }
                )

            identity_ok, identity_detail = self._assert_adapter_identity(
                adapter,
                provider=provider,
                resolved_model=health.resolved_model,
                session=session,
            )
            if not identity_ok:
                return base.model_copy(
                    update={
                        "health_status": StageStatus.PASS,
                        "resolved_through_canonical_resolver": False,
                        "failure_code": ProofFailureCode.PROVIDER_IDENTITY_MISMATCH,
                        "safe_error_excerpt": identity_detail,
                    }
                )

            base = base.model_copy(
                update={
                    "resolved_model": health.resolved_model,
                    "server_model": health.server_model,
                    "server_model_digest": health.server_model_digest,
                    "adapter_class": health.adapter_class,
                    "server_version": health.server_version,
                    "base_url_classification": health.base_url_classification,
                    "health_status": StageStatus.PASS,
                    "resolved_through_canonical_resolver": True,
                    "session_adapter_object_id": str(id(adapter)),
                }
            )

            started = time.perf_counter()
            ok, excerpt, failure, detail = await run_basic_generation(
                adapter,
                provider=provider,
                configured_model=configured_model,
            )
            latency = {"basic_generation_ms": (time.perf_counter() - started) * 1000}
            if not ok:
                return base.model_copy(
                    update={
                        "basic_generation_status": StageStatus.FAIL,
                        "failure_code": failure,
                        "safe_error_excerpt": detail,
                        "latency_ms": latency,
                    }
                )
            base = base.model_copy(
                update={
                    "basic_generation_status": StageStatus.PASS,
                    "basic_generation_excerpt": excerpt,
                    "latency_ms": latency,
                }
            )

            started = time.perf_counter()
            ok, validation, failure, detail = await run_structured_planning(adapter)
            latency = {
                **base.latency_ms,
                "structured_plan_ms": (time.perf_counter() - started) * 1000,
            }
            if not ok:
                return base.model_copy(
                    update={
                        "structured_planning_status": StageStatus.FAIL,
                        "failure_code": failure,
                        "safe_error_excerpt": detail,
                        "latency_ms": latency,
                    }
                )
            base = base.model_copy(
                update={
                    "structured_planning_status": StageStatus.PASS,
                    "planning_validation": validation,
                    "latency_ms": latency,
                }
            )

            started = time.perf_counter()
            ok, mode, failure, detail = await run_tool_call_and_execution(
                adapter,
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
                task_executor=session.task_executor,
                repository=session.repository,
            )
            latency = {
                **base.latency_ms,
                "tool_execution_ms": (time.perf_counter() - started) * 1000,
            }
            if not ok:
                return base.model_copy(
                    update={
                        "tool_call_status": StageStatus.FAIL,
                        "tool_execution_status": StageStatus.FAIL,
                        "failure_code": failure,
                        "safe_error_excerpt": detail,
                        "latency_ms": latency,
                    }
                )
            base = base.model_copy(
                update={
                    "tool_call_status": StageStatus.PASS,
                    "tool_execution_status": StageStatus.PASS,
                    "tool_choice_mode": mode,
                    "latency_ms": latency,
                }
            )

            started = time.perf_counter()
            (
                ok,
                answer,
                citation,
                persisted,
                failure,
                detail,
                http_status,
            ) = await run_grounded_ask(
                session.client,
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
                source_id=fixture.source_id,
                repository=session.repository,
            )
            latency = {
                **base.latency_ms,
                "grounded_ask_ms": (time.perf_counter() - started) * 1000,
            }
            if not ok or not persisted:
                return base.model_copy(
                    update={
                        "grounded_ask_status": StageStatus.FAIL,
                        "citation_status": StageStatus.FAIL,
                        "failure_code": failure,
                        "safe_error_excerpt": detail,
                        "ask_run_persisted": persisted,
                        "http_ask_status_code": http_status,
                        "latency_ms": latency,
                    }
                )
            return base.model_copy(
                update={
                    "grounded_ask_status": StageStatus.PASS,
                    "citation_status": StageStatus.PASS,
                    "ask_answer_excerpt": answer,
                    "citation_excerpt": citation,
                    "citation_source_id": fixture.source_id,
                    "ask_run_persisted": persisted,
                    "http_ask_status_code": http_status,
                    "latency_ms": latency,
                }
            )
        finally:
            session.close()


async def run_model_runtime_proof(
    config: ModelRuntimeProofConfig | None = None,
    *,
    evidence_json: Path | None = None,
    evidence_markdown: Path | None = None,
) -> ModelRuntimeProofResult:
    resolved = config or load_proof_config_from_env()
    runner = ModelRuntimeProofRunner(resolved)
    result = await runner.run()
    print(render_terminal_summary(result))
    if evidence_json is not None and evidence_markdown is not None:
        stale = stale_evidence_notice(evidence_json)
        if stale:
            print(stale)
        write_evidence(result, json_path=evidence_json, markdown_path=evidence_markdown)
    elif evidence_json is not None:
        stale = stale_evidence_notice(evidence_json)
        if stale:
            print(stale)
    return result
