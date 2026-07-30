# © Artur Czarnecki. All rights reserved.

"""Orchestration for LKW Ollama / vLLM model runtime portability proof."""

from __future__ import annotations

import secrets
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

from local_workspace_application.model_runtime_proof.config import (
    ModelRuntimeProofConfig,
    apply_env,
    load_proof_config_from_env,
    materialize_provider_env,
)
from local_workspace_application.model_runtime_proof.contracts import (
    FIXTURE_MARKER,
    FixtureRecord,
    IndexInvarianceResult,
    ModelRuntimeProofResult,
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
)
from local_workspace_application.model_runtime_proof.report import (
    render_terminal_summary,
    write_evidence,
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


class ModelRuntimeProofRunner:
    def __init__(self, config: ModelRuntimeProofConfig) -> None:
        self._config = config
        self._document_store = InMemoryDocumentStore()
        self._data_home: Path | None = None
        self._indexing_count_before = 0

    def _git_commit(self) -> str | None:
        try:
            output = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            return output.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    async def run(self) -> ModelRuntimeProofResult:
        started = datetime.now(UTC)
        proof_id = f"lkw-model-runtime-proof:{secrets.token_hex(6)}"
        result = ModelRuntimeProofResult(
            proof_id=proof_id,
            started_at=started,
            repository_commit=self._git_commit(),
        )

        validation_errors = self._config.validate()
        if validation_errors:
            result = result.model_copy(
                update={
                    "completed_at": datetime.now(UTC),
                    "overall_status": ProofOverallStatus.BLOCKED,
                    "limitations": ("invalid_proof_configuration",),
                }
            )
            return result

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
            self._indexing_count_before = count_indexing_operations(
                index_session.repository,
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
            )
        except Exception as exc:
            index_session.close()
            return result.model_copy(
                update={
                    "completed_at": datetime.now(UTC),
                    "overall_status": ProofOverallStatus.FAIL,
                    "limitations": (f"index_fixture_failed:{exc.__class__.__name__}",),
                }
            )
        finally:
            index_session.close()

        embedding_before = index_identity.embedding
        provider_results: dict[str, ProviderQualificationResult] = {}
        overall_pass = True

        for provider in ("ollama", "vllm"):
            provider_result = await self._qualify_provider(
                provider=provider,  # type: ignore[arg-type]
                fixture=fixture,
                index_identity=index_identity,
            )
            provider_results[provider] = provider_result
            if any(
                status is StageStatus.FAIL
                for status in provider_result.stages.model_dump().values()
            ):
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

        collection_ok, vector_ok, document_ok = compare_index_identity(
            index_identity,
            after_identity,
        )
        embedding_ok = compare_embedding_identity(
            embedding_before, after_identity.embedding
        )
        no_reindex_ok = indexing_after == self._indexing_count_before

        index_invariance = IndexInvarianceResult(
            embedding_identity=StageStatus.PASS if embedding_ok else StageStatus.FAIL,
            collection_identity=StageStatus.PASS if collection_ok else StageStatus.FAIL,
            vector_count=StageStatus.PASS if vector_ok else StageStatus.FAIL,
            no_reindex=StageStatus.PASS if no_reindex_ok else StageStatus.FAIL,
        )
        if any(
            status is StageStatus.FAIL
            for status in index_invariance.model_dump().values()
        ):
            overall_pass = False

        fixture_record = FixtureRecord(
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

        completed = datetime.now(UTC)
        final = result.model_copy(
            update={
                "completed_at": completed,
                "fixture": fixture_record,
                "index_identity": index_identity,
                "embedding_identity_before": embedding_before,
                "embedding_identity_after": after_identity.embedding,
                "provider_results": provider_results,
                "index_invariance": index_invariance,
                "overall_status": ProofOverallStatus.PASS
                if overall_pass
                else ProofOverallStatus.FAIL,
                "limitations": (
                    "exact configured Ollama and vLLM pairs only; not universal model parity",
                    "runtime hot swapping not required or proven",
                ),
            }
        )
        return final

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
            env = materialize_provider_env(provider=provider, config=self._config)
            apply_env(env)
            base_url = (
                env["OLLAMA_HOST"]
                if provider == "ollama"
                else env["INTERGRAX_DEFAULT_VLLM_BASE_URL"]
            )
            adapter = LLMAdapterRegistry.create(
                LLMProvider.OLLAMA if provider == "ollama" else LLMProvider.VLLM,
                model=env["INTERGRAX_LLM_MODEL"],
                base_url=base_url,
            )

            base = base.model_copy(
                update={
                    "resolved_model": health.resolved_model,
                    "server_model": health.server_model,
                    "adapter_class": health.adapter_class,
                    "server_version": health.server_version,
                    "base_url_classification": health.base_url_classification,
                    "health_status": StageStatus.PASS,
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
            ok, answer, citation, persisted, failure, detail = await run_grounded_ask(
                session.ask_service,
                tenant_id=fixture.tenant_id,
                workspace_id=fixture.workspace_id,
                source_id=fixture.source_id,
                repository=session.repository,
            )
            latency = {
                **base.latency_ms,
                "grounded_ask_ms": (time.perf_counter() - started) * 1000,
            }
            if not ok:
                return base.model_copy(
                    update={
                        "grounded_ask_status": StageStatus.FAIL,
                        "citation_status": StageStatus.FAIL,
                        "failure_code": failure,
                        "safe_error_excerpt": detail,
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
    if evidence_json and evidence_markdown:
        write_evidence(result, json_path=evidence_json, markdown_path=evidence_markdown)
    return result
