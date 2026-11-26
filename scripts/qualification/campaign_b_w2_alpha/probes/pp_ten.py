# ruff: noqa: E501, I001
"""PP-TEN probe pack (run R04, check CK-09 TENANT-SCOPE, NT-11..NT-15 + static ERL/AHI scan).

Campaign B-1 alpha, node PT-B01. Diagnostic, non-certifying, READ-ONLY on tracked files.
Code under test: scratch worktree at audited baseline B (258bbf6abb6fdbb92ef0185d058527fbb1e47f04).
Design source: WP-11 PLAN section 4 (CK-09) and 5.6 (R04); finding texts in WP-1 PLAN (S2b, S3, S4), WP-5 PLAN (E3, E4, E5).

Semantics: every test ASSERTS THE SAFE behaviour. PASS = defect REFUTED, FAIL = defect CONFIRMED.
Setup problems (import / construction) are converted to ``pytest.skip("INCONCLUSIVE: ...")``; exceptions raised by the
behaviour under test are the observation (decided per probe, see each docstring). An autouse fixture skips the whole pack
if the product code is not imported from the scratch worktree (the stale main checkout must never be exercised).

Factories / product functions used (paths relative to the worktree, lines at B):
  NT-11  intergrax/compat/langchain/documents.py:176 from_langchain_document;
         intergrax/rag/document_loaders/compat/legacy_runtime_document.py:39 from_legacy_rag_hit
  NT-12  intergrax/runtime/wiring/llm_resolver.py:28,51 consume_routing_evaluation / evaluate_llm_routing (Tier-1);
         intergrax/applications/_shared/llm_resolver.py:39,71 (Tier-3); evaluate_llm_routing is the function every
         resolve_llm_adapter path calls first (runtime/wiring/llm_resolver.py:117, applications/_shared/llm_resolver.py:227)
  NT-13  intergrax/applications/_shared/runtime_boundary_adapters.py:42 application_profile_to_runtime_profile;
         intergrax/applications/_shared/memory_runtime_bridge.py:27 apply_memory_profile_to_runtime_config;
         intergrax/applications/_shared/tenant_quota_wiring.py:24 resolve_tenant_quota_wiring
  NT-14  agents/local_search/steps/search_job.py:176 run_search_job (the step), with
         intergrax/agents/authoring/runtime_tool_helpers.py:35,55 request_metadata / resolve_request_scope and
         intergrax/tools/providers/rag/source_scope_transport.py (scope contextvar read inside the gateway call)
  NT-15  intergrax/debug/app.py:39 create_debug_app (the single factory behind the callers
         applications/lab_application/host/factory.py:164, applications/attestation_demo/host/factory.py:97,
         applications/intergrax_assistant_application/host/factory.py:105, applications/poc_template_application/host/factory.py:94,
         intergrax/applications/_shared/harness_host_auxiliary_wiring.py:119 create_harness_host_debug_app, intergrax/lab/organization_worker.py:124);
         intergrax/applications/_shared/harness_auth.py:107,296,319 require_harness_api_key / HarnessApiKeyMiddleware /
         apply_harness_auth_middleware (the functions hosts use to guard the app)
  E5     intergrax/runtime/enterprise_reliability/{admission_boundary,sqlite_provider_invocation_store}.py,
         intergrax/contracts/{provider_invocation,provider_invocation_store}.py,
         intergrax/runtime/adaptive/policy_learning_approval.py (static scan)
Not built: create_harness_host_debug_app needs a full HarnessHostRuntime; all callers funnel into create_debug_app with the
same keyword shape, which the probe calls with explicit tmp_path stores.

| probe id                                           | findings            | hypothesis                                                              | PREDICTED | reason from static reading |
|----------------------------------------------------|---------------------|-------------------------------------------------------------------------|-----------|----------------------------|
| test_nt11a_from_langchain_document_foreign_tenant  | A8-020 / X-ESC-E3   | converter mints scope.tenant_id from foreign metadata, no caller bind   | FAIL      | documents.py:204 takes tenant_id from metadata; signature has no tenant parameter |
| test_nt11b_from_legacy_rag_hit_foreign_tenant      | A8-020 / X-ESC-E3   | same through the legacy vector-store hit adapter                        | FAIL      | legacy_runtime_document.py:39-70 forwards metadata to from_langchain_document; no tenant parameter |
| test_nt12a_tier1_routing_record_not_shared         | X-A4-016            | module-global routing record read by the other tenant (Tier-1 resolver) | FAIL      | runtime/wiring/llm_resolver.py:20,75 single process-wide _last_routing_evaluation, last writer wins |
| test_nt12b_tier3_routing_record_not_shared         | X-A4-016            | same on the Tier-3 resolver                                             | FAIL      | applications/_shared/llm_resolver.py:31,99 same pattern; consume has no caller (observation) |
| test_nt13a_profile_projection_no_silent_drop       | X-A8-003            | configured retention/scope/fairness/recovery values silently dropped    | FAIL      | host_profile_slices.py:106,117,126 extra="ignore"; runtime_boundary_adapters.py:54-60 model_validate(model_dump) |
| test_nt13b_dropped_fields_still_enforced_elsewhere | X-A8-003            | the dropped values do not reach any enforcement route                   | PASS      | memory_runtime_bridge.py:35-36 and tenant_quota_wiring.py:30 read the Tier-3 profile directly (observation: fairness plan uses placeholder tenants, quota wiring has no non-def caller) |
| test_nt14a_step_metadata_cannot_widen_rag_scope    | X-S2b               | step-authored metadata overrides host-supplied tenant/workspace/sources | PASS      | request_metadata (runtime_tool_helpers.py:35-52) reads only exec_ctx.request.metadata when a request exists; tenant from typed request |
| test_nt14b_step_cannot_author_scope_without_host   | X-S2b               | agent step supplies allowed_source_ids when the host sent none          | PASS      | same; without exec_ctx the job returns before any retrieval (search_job.py:243-250) |
| test_nt15a_debug_app_keyed_rejects_unauthenticated | X-ESC-E4            | debug app registry reachable without credentials when a key is set      | PASS      | router dependency require_harness_api_key (router.py:187-191, harness_auth.py:107-128) |
| test_nt15b_debug_app_experiment_registry_tenant    | X-ESC-E4            | tenant B (valid token) reads tenant A experiments                        | FAIL      | ExperimentRecord/RegisterExperimentRequest have no tenant field; routes router.py:416-470 take no tenant; store has no tenant key |
| test_nt15c_debug_app_callers_static_enumeration    | X-ESC-E4            | strict-capable caller mounts the debug app with no auth wiring           | PASS      | only lab factory references strict mode and it requires/uses harness auth; others enumerate as non-strict (records the list) |
| test_nt15d_debug_router_guard_honours_identity_idp | X-ESC-E4            | IdP-only host: router-level guard lets anonymous callers through        | FAIL      | require_harness_api_key returns when state.resolved_api_key is None (harness_auth.py:119-121), ignoring the identity provider |
| test_nt15e_erl_ahi_tenant_reference_scan           | X-ESC-E5            | provider-invocation record/store carry no tenant identity (+AHI list)   | FAIL      | contracts/provider_invocation.py has no tenant field; sqlite store keyed by invocation_id only (store.py:213-228); AHI approval store keyed by proposal_id (recorded, not asserted) |
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_VENDOR_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "TAVILY_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


def _skip(reason: str) -> Any:
    pytest.skip(f"INCONCLUSIVE: {reason}")
    raise AssertionError("unreachable")


def _worktree_root() -> Path:
    import intergrax

    return Path(intergrax.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _scrub_env_and_check_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("INTERGRAX_", "LAB_")) or key in _VENDOR_ENV:
            monkeypatch.delenv(key, raising=False)
    try:
        root = _worktree_root().as_posix().lower()
    except Exception as exc:  # noqa: BLE001
        _skip(f"intergrax not importable: {type(exc).__name__}: {exc}")
    if "/_wt/" not in root:
        _skip(f"product code is not imported from the scratch worktree: {root}")


# ----------------------------------------------------------------------------------------------------------------------
# NT-11  hostile metadata['tenant_id'] through the converters (A8-020 / ESC-E3)
# ----------------------------------------------------------------------------------------------------------------------

_HOSTILE_TENANT_VALUES = (_TENANT_B, " tenant-b ", "default", "*", "tenant-a​")


def _lc_document(tenant_value: object) -> Any:
    try:
        from langchain_core.documents import Document
    except Exception as exc:  # noqa: BLE001
        _skip(f"langchain_core not importable: {type(exc).__name__}: {exc}")
    return Document(
        id=None,
        page_content="hostile payload",
        metadata={"tenant_id": tenant_value, "source_kind": "file", "source_id": "abc123"},
    )


def _tenant_binding_params(fn: Any) -> list[str]:
    return [name for name in inspect.signature(fn).parameters if "tenant" in name.lower()]


def test_nt11a_from_langchain_document_foreign_tenant() -> None:
    """Safe = the converter never adopts a tenant it was not told to expect.

    If the function exposes a tenant-binding parameter, a mismatching document must be refused. If it exposes none, the
    tenant written into ``KnowledgeDocument.scope`` must not be the one read from the (foreign) metadata.
    Exceptions from the converter are counted as refusals (safe) only for ValueError/ValidationError/Bridge errors.
    """
    try:
        from intergrax.compat.langchain.documents import (
            LangChainDocumentBridgeError,
            from_langchain_document,
        )
        from pydantic import ValidationError
    except Exception as exc:  # noqa: BLE001
        _skip(f"converter not importable: {type(exc).__name__}: {exc}")
    binding = _tenant_binding_params(from_langchain_document)
    adopted: dict[str, str] = {}
    refused: dict[str, str] = {}
    for value in _HOSTILE_TENANT_VALUES:
        kwargs: dict[str, Any] = {"document_id": "file:abc123"}
        if binding:
            kwargs[binding[0]] = _TENANT_A
        try:
            result = from_langchain_document(_lc_document(value), **kwargs)
        except (LangChainDocumentBridgeError, ValidationError, ValueError) as exc:
            refused[repr(value)] = type(exc).__name__
            continue
        except TypeError as exc:
            _skip(f"harness error calling converter: {exc}")
        adopted[repr(value)] = result.scope.tenant_id
    try:
        from_langchain_document(
            _lc_document(None),
            **{"document_id": "file:abc123", **({binding[0]: _TENANT_A} if binding else {})},
        )
        missing = "accepted"
    except Exception as exc:  # noqa: BLE001
        missing = f"refused:{type(exc).__name__}"
    print(
        f"OBSERVATION NT11a: tenant_binding_params={binding} adopted_from_metadata={adopted} "
        f"refused={refused} metadata_without_tenant={missing}"
    )
    foreign_adopted = [k for k, v in adopted.items() if v.strip() != _TENANT_A]
    assert not foreign_adopted, (
        "from_langchain_document minted scope.tenant_id from foreign metadata with no caller-bound tenant"
    )


def test_nt11b_from_legacy_rag_hit_foreign_tenant() -> None:
    """Same criterion through ``from_legacy_rag_hit`` (vector-store hit adapter)."""
    try:
        from intergrax.rag.document_loaders.compat.legacy_runtime_document import from_legacy_rag_hit
        from pydantic import ValidationError
        from intergrax.compat.langchain.documents import LangChainDocumentBridgeError
    except Exception as exc:  # noqa: BLE001
        _skip(f"legacy adapter not importable: {type(exc).__name__}: {exc}")
    binding = _tenant_binding_params(from_legacy_rag_hit)
    adopted: dict[str, str] = {}
    refused: dict[str, str] = {}
    for value in _HOSTILE_TENANT_VALUES:
        hit = SimpleNamespace(
            id="file:abc123",
            content="hostile payload",
            metadata={"tenant_id": value, "source_kind": "file", "source_id": "abc123"},
        )
        kwargs: dict[str, Any] = {}
        if binding:
            kwargs[binding[0]] = _TENANT_A
        try:
            result = from_legacy_rag_hit(hit, **kwargs)
        except (LangChainDocumentBridgeError, ValidationError, ValueError) as exc:
            refused[repr(value)] = type(exc).__name__
            continue
        except TypeError as exc:
            _skip(f"harness error calling adapter: {exc}")
        adopted[repr(value)] = result.scope.tenant_id
    print(f"OBSERVATION NT11b: tenant_binding_params={binding} adopted_from_metadata={adopted} refused={refused}")
    foreign_adopted = [k for k, v in adopted.items() if v.strip() != _TENANT_A]
    assert not foreign_adopted, (
        "from_legacy_rag_hit minted scope.tenant_id from foreign hit metadata with no caller-bound tenant"
    )


# ----------------------------------------------------------------------------------------------------------------------
# NT-12  two-tenant routing-evaluation record sharing (X-A4-016)
# ----------------------------------------------------------------------------------------------------------------------


def _routing_profiles() -> tuple[Any, Any]:
    """Return (LLMProfile default/a/b namespace, LLMRoutingProfile) built with the real contracts (no adapter created)."""
    try:
        from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
        from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
        from intergrax.llm_adapters.contracts.routing_profile import LLMRoutingProfile
        from intergrax.llm_adapters.routing.builtin_rules import TenantIdInRule
    except Exception as exc:  # noqa: BLE001
        _skip(f"routing contracts not importable: {type(exc).__name__}: {exc}")
    try:
        default = LLMProfile(provider=LLMProvider.OLLAMA, model="model-default")
        profile_a = LLMProfile(provider=LLMProvider.OLLAMA, model="model-a")
        profile_b = LLMProfile(provider=LLMProvider.OLLAMA, model="model-b")
        routing = LLMRoutingProfile(
            default_profile=default,
            allowed_profiles=(default, profile_a, profile_b),
            rules=(
                TenantIdInRule(tenant_ids=(_TENANT_A,), profile=profile_a),
                TenantIdInRule(tenant_ids=(_TENANT_B,), profile=profile_b),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot build routing profile: {type(exc).__name__}: {exc}")
    return SimpleNamespace(default=default, a=profile_a, b=profile_b), routing


def _interleaved_routing(module: Any, env: Any) -> dict[str, Any]:
    """Tenant A evaluates, tenant B evaluates, then A consumes, then B consumes (events, no sleeps)."""
    from intergrax.llm_adapters.routing import RoutingContext

    module.consume_routing_evaluation()  # clear leftovers from other imports
    a_evaluated, b_evaluated, a_consumed = threading.Event(), threading.Event(), threading.Event()
    out: dict[str, Any] = {"errors": []}

    def worker_a() -> None:
        try:
            module.evaluate_llm_routing(env, routing_context=RoutingContext(tenant_id=_TENANT_A))
            a_evaluated.set()
            b_evaluated.wait(2.0)
            out["a"] = module.consume_routing_evaluation()
        except BaseException as exc:  # noqa: BLE001
            out["errors"].append(f"a:{type(exc).__name__}:{exc}")
        finally:
            a_evaluated.set()
            a_consumed.set()

    def worker_b() -> None:
        try:
            a_evaluated.wait(2.0)
            module.evaluate_llm_routing(env, routing_context=RoutingContext(tenant_id=_TENANT_B))
            b_evaluated.set()
            a_consumed.wait(2.0)
            out["b"] = module.consume_routing_evaluation()
        except BaseException as exc:  # noqa: BLE001
            out["errors"].append(f"b:{type(exc).__name__}:{exc}")
        finally:
            b_evaluated.set()

    threads = [threading.Thread(target=worker_a), threading.Thread(target=worker_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    return out


def _judge_routing(label: str, out: dict[str, Any], extra: str = "") -> None:
    if out["errors"]:
        _skip(f"worker error in {label}: {out['errors']}")

    def model_of(item: Any) -> str | None:
        return None if item is None else item.selected_profile.model

    got_a, got_b = model_of(out.get("a")), model_of(out.get("b"))
    print(
        f"OBSERVATION NT12{label}: tenant_a_consumed={got_a} tenant_b_consumed={got_b} "
        f"(expected own record or None; a_lost={got_a is None} b_lost={got_b is None}){extra}"
    )
    assert got_a in (None, "model-a"), f"tenant A consumed the routing record of another tenant ({got_a})"
    assert got_b in (None, "model-b"), f"tenant B consumed the routing record of another tenant ({got_b})"


def test_nt12a_tier1_routing_record_not_shared() -> None:
    try:
        from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
        from intergrax.runtime.wiring import llm_resolver as module
    except Exception as exc:  # noqa: BLE001
        _skip(f"Tier-1 resolver not importable: {type(exc).__name__}: {exc}")
    profiles, routing = _routing_profiles()
    try:
        env = RuntimeEnvironmentProfile(
            profile_id="nt12.tier1", llm_profile=profiles.default, llm_routing_profile=routing
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot build RuntimeEnvironmentProfile: {type(exc).__name__}: {exc}")
    _judge_routing("a", _interleaved_routing(module, env))


def _consume_callers(root: Path) -> list[str]:
    hits: list[str] = []
    for base in ("intergrax", "applications", "agents"):
        for path in (root / base).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            if "consume_routing_evaluation" in text and "def consume_routing_evaluation" not in text:
                hits.append(path.relative_to(root).as_posix())
    return sorted(hits)


def test_nt12b_tier3_routing_record_not_shared() -> None:
    try:
        from intergrax.applications._shared import llm_resolver as module
        from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    except Exception as exc:  # noqa: BLE001
        _skip(f"Tier-3 resolver not importable: {type(exc).__name__}: {exc}")
    profiles, routing = _routing_profiles()
    try:
        env = ApplicationEnvironmentProfile.lab_defaults(profile_id="nt12.tier3")
        env.llm_profile = profiles.default
        env.llm_routing_profile = routing
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot build ApplicationEnvironmentProfile: {type(exc).__name__}: {exc}")
    callers = [c for c in _consume_callers(_worktree_root())]
    extra = f" files_referencing_consume_routing_evaluation={callers}"
    _judge_routing("b", _interleaved_routing(module, env), extra=extra)


# ----------------------------------------------------------------------------------------------------------------------
# NT-13  projection of retention / scope / fairness / recovery fields (X-A8-003)
# ----------------------------------------------------------------------------------------------------------------------


def _t3_profiles() -> Any:
    try:
        from intergrax.applications.contracts.application_recovery_contract import (
            ApplicationRecoveryContract,
        )
        from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
        from intergrax.applications.contracts.environment_profile import sub_profiles as t3
        from intergrax.contracts import host_profile_slices as t0
    except Exception as exc:  # noqa: BLE001
        _skip(f"profile modules not importable: {type(exc).__name__}: {exc}")
    return SimpleNamespace(Env=ApplicationEnvironmentProfile, t3=t3, t0=t0, Recovery=ApplicationRecoveryContract)


def _configured_env(p: Any) -> Any:
    try:
        env = p.Env.lab_defaults(profile_id="nt13.configured")
        env.memory_profile = p.t3.MemoryProfile(retention_days=7, scope_boundary="workspace")
        env.cost_profile = p.t3.CostProfile(
            tenant_fairness_quotas_enabled=True, max_tool_calls=11, budget_enforcement_enabled=False
        )
        env.reliability_profile = p.t3.ReliabilityProfile(
            compensation_enabled=True, recovery_contract=p.Recovery(max_resume_attempts=5)
        )
        return env
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot configure Tier-3 profile: {type(exc).__name__}: {exc}")


_NT13_CONFIGURED = (
    ("memory_profile", "retention_days", 7),
    ("memory_profile", "scope_boundary", "workspace"),
    ("cost_profile", "tenant_fairness_quotas_enabled", True),
    ("cost_profile", "max_tool_calls", 11),
    ("cost_profile", "budget_enforcement_enabled", False),
    ("reliability_profile", "compensation_enabled", True),
    ("reliability_profile", "recovery_contract", "present"),
)


def test_nt13a_profile_projection_no_silent_drop() -> None:
    """Safe = each configured Tier-3-only value either reaches the runtime slice or the projection refuses.

    An exception from the projection counts as a refusal (fail-on-unmapped = safe) for ValueError/ValidationError.
    """
    p = _t3_profiles()
    env = _configured_env(p)
    try:
        from intergrax.applications._shared.runtime_boundary_adapters import (
            application_profile_to_runtime_profile,
        )
        from pydantic import ValidationError
    except Exception as exc:  # noqa: BLE001
        _skip(f"adapter not importable: {type(exc).__name__}: {exc}")
    extra_fields = {
        name: sorted(set(getattr(p.t3, cls).model_fields) - set(getattr(p.t0, cls).model_fields))
        for name, cls in (("memory", "MemoryProfile"), ("cost", "CostProfile"), ("reliability", "ReliabilityProfile"))
    }
    try:
        runtime_profile = application_profile_to_runtime_profile(env)
    except (ValueError, ValidationError) as exc:
        print(
            f"OBSERVATION NT13a: projection refused ({type(exc).__name__}); tier3_only_field_counts="
            f"{ {k: len(v) for k, v in extra_fields.items()} }"
        )
        return
    except Exception as exc:  # noqa: BLE001
        _skip(f"harness error in projection: {type(exc).__name__}: {exc}")
    assert runtime_profile is not None
    dropped: list[str] = []
    carried: list[str] = []
    for slice_name, field_name, expected in _NT13_CONFIGURED:
        projected = getattr(runtime_profile, slice_name).model_dump(mode="python")
        if field_name in projected and (expected == "present" or projected[field_name] == expected):
            carried.append(f"{slice_name}.{field_name}")
        else:
            dropped.append(f"{slice_name}.{field_name}")
    print(
        f"OBSERVATION NT13a: dropped={dropped} carried={carried} "
        f"tier3_only_field_counts={ {k: len(v) for k, v in extra_fields.items()} }"
    )
    assert not dropped, f"projection silently dropped configured values: {dropped}"


def test_nt13b_dropped_fields_still_enforced_elsewhere() -> None:
    """Context for NT-13a: do the same configured values reach enforcement through the Tier-3 bridges?"""
    p = _t3_profiles()
    env = _configured_env(p)
    try:
        from intergrax.applications._shared.memory_runtime_bridge import apply_memory_profile_to_runtime_config
        from intergrax.applications._shared.tenant_quota_wiring import resolve_tenant_quota_wiring
        from intergrax.applications.contracts.application_host import ApplicationProfile
        from intergrax.runtime.execution.host_runtime_config import RuntimeConfig
        from testing_support.builder import FakeLLMAdapter
    except Exception as exc:  # noqa: BLE001
        _skip(f"bridges not importable: {type(exc).__name__}: {exc}")
    try:
        config = apply_memory_profile_to_runtime_config(RuntimeConfig(llm_adapter=FakeLLMAdapter()), env.memory_profile)
        env.application_profile = ApplicationProfile.PRODUCT
        quota = resolve_tenant_quota_wiring(env)
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot exercise bridges: {type(exc).__name__}: {exc}")
    plan_tenants = sorted({getattr(q, "tenant_id", None) for q in (quota.plan.quotas if quota.plan else [])} - {None})
    print(
        f"OBSERVATION NT13b: memory_retention_days={config.memory_retention_days} "
        f"memory_scope_boundary={config.memory_scope_boundary} fairness_enabled={quota.enabled} "
        f"fairness_plan_tenants={plan_tenants} (placeholder-looking tenant ids are not asserted) "
        f"recovery_contract_consumers=reliability_wiring.py,recovery_contract_wiring.py (static, not exercised)"
    )
    assert config.memory_retention_days == 7, "retention_days not applied to runtime config"
    assert config.memory_scope_boundary == "workspace", "scope_boundary not applied to runtime config"
    assert quota.enabled is True, "tenant_fairness_quotas_enabled not honoured by the quota wiring"


# ----------------------------------------------------------------------------------------------------------------------
# NT-14  agent step widening the RAG scope (X-S2b)
# ----------------------------------------------------------------------------------------------------------------------


def _search_kit() -> Any:
    try:
        from intergrax.contracts.agent_step_context import AgentStepContext
        from intergrax.contracts.tool_request import ToolResponse, ToolResponseStatus
        from intergrax.tools.providers.rag.source_scope_transport import (
            current_rag_retrieval_source_scope,
        )
        from local_search.diagnostics import SearchSummaryReason
        from local_search.steps.search_job import run_search_job
        from testing_support.builder import (
            build_runtime_execution_context_for_tests,
            build_runtime_request_for_tests,
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"local_search step not importable: {type(exc).__name__}: {exc}")
    return SimpleNamespace(
        Step=AgentStepContext,
        Response=ToolResponse,
        Status=ToolResponseStatus,
        scope_now=current_rag_retrieval_source_scope,
        Reason=SearchSummaryReason,
        run=run_search_job,
        mk_ctx=build_runtime_execution_context_for_tests,
        mk_req=build_runtime_request_for_tests,
    )


def _recording_gateway(kit: Any, seen: list[dict[str, Any]]) -> Any:
    async def _invoke(request: Any) -> Any:
        scope = kit.scope_now()
        seen.append(
            {
                "tool_input": dict(request.input or {}),
                "scope_present": scope.is_present,
                "allowed_source_ids": tuple(scope.allowed_source_ids),
            }
        )
        return kit.Response(
            request_id=request.request_id,
            status=kit.Status.SUCCESS,
            output={
                "used": True,
                "chunks": [{"id": "c1", "text": "evidence", "score": 0.9, "metadata": {"source_path": "/d/x.txt"}}],
                "citations": [],
                "context_text": "evidence",
                "reason": "",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke)
    return gateway


def _hostile_step_metadata() -> dict[str, Any]:
    return {
        "allowed_source_ids": ["src-1", "src-2", "src-other-tenant"],
        "workspace_id": "ws-other",
        "tenant_id": _TENANT_B,
        "collection_id": "col-other",
        "query": "hostile query",
        "top_k": 500,
        "user_id": "user-hostile",
    }


async def test_nt14a_step_metadata_cannot_widen_rag_scope() -> None:
    kit = _search_kit()
    seen: list[dict[str, Any]] = []
    try:
        request = kit.mk_req(
            seed="nt14a",
            agent_id="local_search",
            tenant_id=_TENANT_A,
            user_id="user-a",
            session_id="s1",
            message="",
            metadata={"query": "host query", "workspace_id": "ws-host", "top_k": 3, "allowed_source_ids": ["src-1"]},
        )
        exec_ctx = kit.mk_ctx(
            seed="nt14a",
            agent_id="local_search",
            request=request,
            tool_gateway=_recording_gateway(kit, seen),
            tenant_id=_TENANT_A,
        )
        step = kit.Step(
            tenant_id=_TENANT_B,
            run_id=str(exec_ctx.run_id),
            agent_id="local_search",
            contract_id="local_search",
            metadata={"uaep_exec_ctx": exec_ctx, **_hostile_step_metadata()},
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot build step fixture: {type(exc).__name__}: {exc}")
    output = await kit.run(step)
    summary = output.get("search_summary", {}) if isinstance(output, dict) else {}
    print(
        f"OBSERVATION NT14a: used={summary.get('used')} reason={summary.get('reason')} gateway_calls={len(seen)} "
        f"seen={seen[:1]}"
    )
    if not seen:
        _skip(f"retrieval gateway never invoked (reason={summary.get('reason')})")
    call = seen[0]
    assert call["tool_input"].get("tenant_id") == _TENANT_A, "step metadata changed the tenant of the retrieval call"
    assert call["tool_input"].get("workspace_id") == "ws-host", (
        "step metadata changed the workspace of the retrieval call"
    )
    assert call["tool_input"].get("top_k") == 3, "step metadata changed top_k of the retrieval call"
    assert call["tool_input"].get("query") == "host query", "step metadata replaced the host query"
    assert call["scope_present"] and call["allowed_source_ids"] == ("src-1",), (
        f"step metadata widened the source scope: {call['allowed_source_ids']}"
    )


async def test_nt14b_step_cannot_author_scope_without_host() -> None:
    kit = _search_kit()
    seen: list[dict[str, Any]] = []
    try:
        request = kit.mk_req(
            seed="nt14b",
            agent_id="local_search",
            tenant_id=_TENANT_A,
            user_id="user-a",
            session_id="s1",
            message="",
            metadata={"query": "host query", "workspace_id": "ws-host"},
        )
        exec_ctx = kit.mk_ctx(
            seed="nt14b",
            agent_id="local_search",
            request=request,
            tool_gateway=_recording_gateway(kit, seen),
            tenant_id=_TENANT_A,
        )
        step_with_ctx = kit.Step(
            tenant_id=_TENANT_A,
            run_id=str(exec_ctx.run_id),
            agent_id="local_search",
            contract_id="local_search",
            metadata={"uaep_exec_ctx": exec_ctx, "allowed_source_ids": ["src-x"]},
        )
        step_without_ctx = kit.Step(
            tenant_id=_TENANT_A,
            run_id=str(exec_ctx.run_id),
            agent_id="local_search",
            contract_id="local_search",
            message="hostile orphan query",
            metadata=_hostile_step_metadata(),
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot build step fixture: {type(exc).__name__}: {exc}")
    out_ctx = await kit.run(step_with_ctx)
    out_orphan = await kit.run(step_without_ctx)
    summary_ctx = out_ctx.get("search_summary", {}) if isinstance(out_ctx, dict) else {}
    summary_orphan = out_orphan.get("search_summary", {}) if isinstance(out_orphan, dict) else {}
    print(
        f"OBSERVATION NT14b: host_sent_no_scope -> gateway_calls={len(seen)} scope_present="
        f"{seen[0]['scope_present'] if seen else 'n/a'} allowed={seen[0]['allowed_source_ids'] if seen else 'n/a'} "
        f"(absent scope = unrestricted, by design of the transport); no_exec_ctx -> used={summary_orphan.get('used')} "
        f"reason={summary_orphan.get('reason')} summary_ctx_reason={summary_ctx.get('reason')}"
    )
    if not seen:
        _skip(f"retrieval gateway never invoked (reason={summary_ctx.get('reason')})")
    assert not seen[0]["scope_present"], (
        f"agent-authored allowed_source_ids became the retrieval scope: {seen[0]['allowed_source_ids']}"
    )
    assert summary_orphan.get("used") is False, "step without execution context performed retrieval from step metadata"
    assert summary_orphan.get("reason") == kit.Reason.TOOL_GATEWAY_NOT_AVAILABLE.value


# ----------------------------------------------------------------------------------------------------------------------
# NT-15  STRICT hosts and create_debug_app (X-ESC-E4)
# ----------------------------------------------------------------------------------------------------------------------

_EXPERIMENT_BODY = {
    "hypothesis": "nt15 probe hypothesis",
    "capability": "nt15.capability",
    "notes": "owner tenant-a",
}


def _debug_app(tmp_path: Path) -> Any:
    try:
        from intergrax.debug.app import create_debug_app
    except Exception as exc:  # noqa: BLE001
        _skip(f"debug app not importable: {type(exc).__name__}: {exc}")
    try:
        return create_debug_app(
            db_path=tmp_path / "trace.db",
            experiments_db_path=tmp_path / "experiments.db",
            runtime_events_db_path=tmp_path / "runtime_events.db",
            checkpoints_db_path=tmp_path / "checkpoints.db",
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot build debug app: {type(exc).__name__}: {exc}")


def _client(app: Any) -> Any:
    try:
        from fastapi.testclient import TestClient

        return TestClient(app)
    except Exception as exc:  # noqa: BLE001
        _skip(f"TestClient unavailable: {type(exc).__name__}: {exc}")


class _StubIdentityProvider:
    """In-process identity provider: bearer tokens ``token-a`` / ``token-b`` map to tenants A and B."""

    def __init__(self) -> None:
        from intergrax.integrations.contracts.identity_provider import IdentityUser

        self._users = {
            "token-a": IdentityUser(user_id="user-a", tenant_id=_TENANT_A),
            "token-b": IdentityUser(user_id="user-b", tenant_id=_TENANT_B),
        }

    def verify_token(self, token: str) -> Any:
        return self._users[token]

    def userinfo(self, token: str) -> Any:
        return self._users[token]

    def list_tenants(self, *, limit: int = 50) -> list[Any]:
        return []


def test_nt15a_debug_app_keyed_rejects_unauthenticated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A STRICT host must have a harness key (lab factory.py:88 enforces it); then the registry must not answer anonymously."""
    key = "k" * 32
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", key)
    client = _client(_debug_app(tmp_path))
    anon = client.post("/debug/experiments", json=_EXPERIMENT_BODY)
    wrong = client.post("/debug/experiments", json=_EXPERIMENT_BODY, headers={"X-Api-Key": "w" * 32})
    anon_list = client.get("/debug/experiments")
    ok = client.post("/debug/experiments", json=_EXPERIMENT_BODY, headers={"X-Api-Key": key})
    print(
        f"OBSERVATION NT15a: anonymous_post={anon.status_code} wrong_key_post={wrong.status_code} "
        f"anonymous_list={anon_list.status_code} valid_key_post={ok.status_code}"
    )
    assert anon.status_code in (401, 403), (
        "anonymous write to the experiment registry accepted while a key is configured"
    )
    assert wrong.status_code in (401, 403), "wrong key accepted by the debug experiment registry"
    assert anon_list.status_code in (401, 403), (
        "anonymous read of the experiment registry accepted while a key is configured"
    )


def test_nt15b_debug_app_experiment_registry_tenant(tmp_path: Path) -> None:
    """Multi-tenant host shape: identity provider + the host auth middleware (``apply_harness_auth_middleware``).

    Safe = a valid tenant-B token cannot read the experiment that a tenant-A token registered (registry tenant-scoped or
    routes unmounted). The middleware must still keep anonymous callers out (checked, reported).
    """
    try:
        from intergrax.applications._shared.harness_auth import HarnessAuthState, apply_harness_auth_middleware
    except Exception as exc:  # noqa: BLE001
        _skip(f"harness auth not importable: {type(exc).__name__}: {exc}")
    app = _debug_app(tmp_path)
    try:
        app.state.harness_auth = HarnessAuthState(identity_provider=_StubIdentityProvider(), tenant_required=True)
        apply_harness_auth_middleware(app, require_auth=True)
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot wire identity auth: {type(exc).__name__}: {exc}")
    client = _client(app)
    anon = client.get("/debug/experiments")
    created = client.post("/debug/experiments", json=_EXPERIMENT_BODY, headers={"Authorization": "Bearer token-a"})
    if created.status_code != 201:
        _skip(f"tenant-A registration not accepted: {created.status_code} {created.text[:120]}")
    experiment_id = created.json().get("experiment_id")
    listed_b = client.get("/debug/experiments", headers={"Authorization": "Bearer token-b"})
    shown_b = client.get(f"/debug/experiments/{experiment_id}", headers={"Authorization": "Bearer token-b"})
    ids_seen_by_b = (
        [e.get("experiment_id") for e in listed_b.json().get("experiments", [])] if listed_b.status_code == 200 else []
    )
    try:
        from intergrax.experiments.models import ExperimentRecord, RegisterExperimentRequest

        tenant_fields = {
            "ExperimentRecord": [f for f in ExperimentRecord.model_fields if "tenant" in f],
            "RegisterExperimentRequest": [f for f in RegisterExperimentRequest.model_fields if "tenant" in f],
        }
    except Exception:  # noqa: BLE001
        tenant_fields = {}
    print(
        f"OBSERVATION NT15b: anonymous_list_status={anon.status_code} tenant_a_register={created.status_code} "
        f"tenant_b_list_status={listed_b.status_code} tenant_b_sees_a_experiment={experiment_id in ids_seen_by_b} "
        f"tenant_b_show_status={shown_b.status_code} tenant_fields={tenant_fields}"
    )
    assert experiment_id not in ids_seen_by_b, "tenant B listed an experiment registered by tenant A"
    assert shown_b.status_code in (401, 403, 404), (
        f"tenant B read tenant A's experiment by id (status {shown_b.status_code})"
    )


def test_nt15c_debug_app_callers_static_enumeration() -> None:
    """Static: enumerate every module that calls ``create_debug_app`` / ``create_harness_host_debug_app``.

    Records for each caller whether it is strict-capable (mentions STRICT / production_mode / a harness-key requirement)
    and whether it references any harness auth wiring. FAIL only on a concrete path: a strict-capable caller module with
    no auth-wiring reference at all.
    """
    root = _worktree_root()
    call_re = re.compile(r"(?<!def )\b(create_debug_app|create_harness_host_debug_app)\s*\(")
    strict_re = re.compile(r"ExecutionMode\.STRICT|production_mode|requires_harness_api_key|\bSTRICT\b")
    guard_re = re.compile(
        r"HarnessAuthState|harness_auth|require_harness_auth|resolve_harness_api_key|apply_harness_auth_middleware"
        r"|wire_application_identity|INTERGRAX_HARNESS_API_KEY|require_agent_platform_admin_auth"
    )
    rows: list[dict[str, Any]] = []
    for base in ("intergrax", "applications", "agents"):
        for path in (root / base).rglob("*.py"):
            parts = set(path.parts)
            if "__pycache__" in parts or "tests" in parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            if not call_re.search(text):
                continue
            rows.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "strict_capable": bool(strict_re.search(text)),
                    "auth_wiring": bool(guard_re.search(text)),
                }
            )
    rows.sort(key=lambda r: r["file"])
    print(
        "OBSERVATION NT15c: callers="
        + "; ".join(f"{r['file']}[strict={r['strict_capable']},auth={r['auth_wiring']}]" for r in rows)
    )
    if not rows:
        _skip("no callers enumerated (path layout differs?)")
    offenders = [r["file"] for r in rows if r["strict_capable"] and not r["auth_wiring"]]
    assert not offenders, f"strict-capable host mounts the debug app without any auth wiring: {offenders}"


def test_nt15d_debug_router_guard_honours_identity_idp(tmp_path: Path) -> None:
    """Router-level guard only (no host middleware): identity-provider-only auth state, anonymous caller.

    Safe = the router's own dependency (``require_harness_api_key``) denies an anonymous caller when the host configured an
    identity provider and no static key. Read-only GET; nothing is written.
    """
    try:
        from intergrax.applications._shared.harness_auth import HarnessAuthState
    except Exception as exc:  # noqa: BLE001
        _skip(f"harness auth not importable: {type(exc).__name__}: {exc}")
    app = _debug_app(tmp_path)
    try:
        app.state.harness_auth = HarnessAuthState(identity_provider=_StubIdentityProvider(), tenant_required=True)
    except Exception as exc:  # noqa: BLE001
        _skip(f"cannot set auth state: {type(exc).__name__}: {exc}")
    client = _client(app)
    anon = client.get("/debug/experiments")
    token = client.get("/debug/experiments", headers={"Authorization": "Bearer token-a"})
    print(f"OBSERVATION NT15d: anonymous_list={anon.status_code} tenant_a_token_list={token.status_code}")
    assert anon.status_code in (401, 403), (
        "debug router answered an anonymous caller although an identity provider is configured"
    )


def _tenant_ref_counts(root: Path, rel_paths: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel in rel_paths:
        path = root / rel
        counts[rel] = (
            len(re.findall(r"tenant", path.read_text(encoding="utf-8"), re.IGNORECASE)) if path.exists() else -1
        )
    return counts


def _classes_with_method(root: Path, method: str) -> list[str]:
    found: list[str] = []
    for path in (root / "intergrax").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if f"def {method}" not in text:
                continue
            tree = ast.parse(text)
        except Exception:  # noqa: BLE001
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(item, ast.FunctionDef) and item.name == method for item in node.body
            ):
                found.append(f"{path.relative_to(root).as_posix()}::{node.name}")
    return sorted(found)


def test_nt15e_erl_ahi_tenant_reference_scan() -> None:
    """Static ERL/AHI tenant-reference scan (X-ESC-E5).

    Always records the enumeration (tenant-reference counts per ERL file named by WP-5 E5, AHI approval-store shape).
    FAIL only on a concrete untenanted reference path: the provider-invocation record has no tenant-bearing field AND the
    store read methods take no tenant parameter AND the SQLite schema has no tenant column.
    """
    root = _worktree_root()
    try:
        from intergrax.contracts.provider_invocation import ProviderInvocation, ProviderInvocationOutcome
        from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
            SQLiteProviderInvocationStore,
        )
    except Exception as exc:  # noqa: BLE001
        _skip(f"provider invocation contracts not importable: {type(exc).__name__}: {exc}")
    named = [
        "intergrax/runtime/enterprise_reliability/admission_boundary.py",
        "intergrax/contracts/enterprise_reliability/admission_boundary.py",
        "intergrax/runtime/enterprise_reliability/sqlite_provider_invocation_store.py",
        "intergrax/contracts/provider_invocation.py",
        "intergrax/contracts/provider_invocation_store.py",
        "intergrax/runtime/adaptive/policy_learning_approval.py",
    ]
    counts = _tenant_ref_counts(root, named)
    record_tenant_fields = [f for f in ProviderInvocation.model_fields if "tenant" in f.lower()] + [
        f for f in ProviderInvocationOutcome.model_fields if "tenant" in f.lower()
    ]
    read_methods = {
        name: [
            p
            for p in inspect.signature(getattr(SQLiteProviderInvocationStore, name)).parameters
            if "tenant" in p.lower()
        ]
        for name in ("get_invocation", "get_outcome", "put_invocation", "put_outcome", "insert_invocation_if_absent")
    }
    store_source = (root / "intergrax/runtime/enterprise_reliability/sqlite_provider_invocation_store.py").read_text(
        encoding="utf-8"
    )
    schema_block = store_source.split("def _create_schema_v1", 1)[-1].split("def _persist_uncertain_write_marker", 1)[0]
    schema_has_tenant = "tenant" in schema_block.lower()
    ahi_impls = _classes_with_method(root, "record_approval")
    ahi_source = (root / "intergrax/runtime/adaptive/policy_learning_approval.py").read_text(encoding="utf-8")
    ahi_tenant_param = bool(re.search(r"def (is_approved|record_approval)\([^)]*tenant", ahi_source))
    print(
        f"OBSERVATION NT15e: tenant_reference_counts={counts} record_tenant_fields={record_tenant_fields} "
        f"store_method_tenant_params={read_methods} sqlite_schema_has_tenant_column={schema_has_tenant} "
        f"ahi_approval_store_implementations={ahi_impls} ahi_approval_methods_take_tenant={ahi_tenant_param}"
    )
    untenanted_path = not record_tenant_fields and not any(read_methods.values()) and not schema_has_tenant
    assert not untenanted_path, (
        "provider-invocation record, store API and SQLite schema carry no tenant identity (ERL tenant continuity not representable)"
    )
