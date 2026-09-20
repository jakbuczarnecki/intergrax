# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X2B — canonical host override one-resolution enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticCompositionError,
    DiagnosticCompositionOverrides,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_scope_discovery_service,
    resolve_host_diagnostic_read_dependencies,
    resolve_host_diagnostic_read_service,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
    resolve_host_terminal_execution_diagnostic_trigger,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    close_harness_host_runtime,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.applications._shared.test_obs_diag_x2a_canonical_host_diagnostic_composition import (
    _build_product_host as _build_product_host_untyped,
)
from tests.unit.applications._shared.test_obs_diag_x2_diagnostic_composition_pluginability import (
    _CustomGroupingStrategy,
    _RecordingOccurrencePersistence,
    _RecordingProblemPersistence,
    _RecordingReconstructionReader,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _build_product_host(
    tmp_path: Path,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> HarnessHostRuntime:
    return _build_product_host_untyped(tmp_path, overrides=overrides)  # type: ignore[return-value]


@pytest.fixture
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def _install_persistence_resolution_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    import intergrax.applications._shared.diagnostic_read_wiring as read_wiring_module

    counter = {"count": 0}
    original = read_wiring_module.resolve_diagnostic_persistence_composition

    def _counting(*args: object, **kwargs: object) -> object:
        counter["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        read_wiring_module,
        "resolve_diagnostic_persistence_composition",
        _counting,
    )
    return counter


def _exercise_host_bound_consumers(runtime: HarnessHostRuntime) -> None:
    resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
        runtime=runtime,
    )
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    resolve_host_diagnostic_read_service(runtime)
    resolve_host_terminal_execution_diagnostic_trigger(runtime)
    build_diagnostic_scope_discovery_service(read_deps)


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        (None, "default"),
        (
            DiagnosticCompositionOverrides(
                problem_persistence=_RecordingProblemPersistence(),
                occurrence_persistence=_RecordingOccurrencePersistence(),
                causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
                execution_reconstruction_reader=_RecordingReconstructionReader(),
                additional_grouping_strategies=(_CustomGroupingStrategy(),),
            ),
            "full",
        ),
        (
            DiagnosticCompositionOverrides(execution_reconstruction_reader=_RecordingReconstructionReader()),
            "reader_only",
        ),
        (
            DiagnosticCompositionOverrides(additional_grouping_strategies=(_CustomGroupingStrategy(),)),
            "grouping_only",
        ),
        (
            DiagnosticCompositionOverrides(problem_persistence=_RecordingProblemPersistence()),
            "partial_persistence",
        ),
    ],
)
def test_x2b_persistence_resolution_count_one_per_host_configuration(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
    overrides: DiagnosticCompositionOverrides | None,
    label: str,
) -> None:
    del label
    counter = _install_persistence_resolution_counter(monkeypatch)
    runtime = _build_product_host(tmp_path, overrides=overrides)
    _exercise_host_bound_consumers(runtime)
    assert counter["count"] == 1


def test_x2b_write_read_persistence_object_identity_with_overrides(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    custom_problem = _RecordingProblemPersistence()
    custom_occurrence = _RecordingOccurrencePersistence()
    custom_causal = InMemoryCausalEvidencePersistence()
    overrides = DiagnosticCompositionOverrides(
        problem_persistence=custom_problem,
        occurrence_persistence=custom_occurrence,
        causal_evidence_persistence=custom_causal,
    )
    runtime = _build_product_host(tmp_path, overrides=overrides)
    write_deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
        runtime=runtime,
    )
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    assert write_deps is not None
    assert write_deps.persistence.problem_persistence is read_deps.problem_persistence
    assert write_deps.persistence.occurrence_persistence is read_deps.occurrence_persistence
    assert write_deps.persistence.causal_evidence_persistence is read_deps.causal_evidence_persistence


def test_x2b_post_build_conflicting_override_fail_closed(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    runtime = _build_product_host(tmp_path)
    conflicting = DiagnosticCompositionOverrides(
        problem_persistence=_RecordingProblemPersistence(),
    )
    with pytest.raises(DiagnosticCompositionError, match="frozen at host construction"):
        resolve_host_diagnostic_read_dependencies(runtime, overrides=conflicting)


def test_x2b_explicit_host_config_override_uses_stored_bundle(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _install_persistence_resolution_counter(monkeypatch)
    reader = _RecordingReconstructionReader()
    overrides = DiagnosticCompositionOverrides(execution_reconstruction_reader=reader)
    runtime = _build_product_host(tmp_path, overrides=overrides)
    host_config = runtime.env_wiring.composition.diagnostic_composition_overrides
    assert host_config is not None
    stored = resolve_host_diagnostic_read_dependencies(runtime, overrides=host_config)
    assert stored is runtime.host_diagnostic_dependencies
    assert counter["count"] == 1


def test_x2b_borrowed_close_count_zero(tmp_path: Path, _stub_llm: None) -> None:
    custom_problem = _RecordingProblemPersistence()
    custom_occurrence = _RecordingOccurrencePersistence()
    overrides = DiagnosticCompositionOverrides(
        problem_persistence=custom_problem,
        occurrence_persistence=custom_occurrence,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    runtime = _build_product_host(tmp_path, overrides=overrides)
    _exercise_host_bound_consumers(runtime)
    close_harness_host_runtime(runtime)
    assert custom_problem.closed is False
    assert custom_occurrence.closed is False


def test_x2b_host_created_default_persistence_closed_once(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intergrax.applications._shared.harness_host_runtime as host_runtime_module

    closed = {"count": 0}

    def _record_close(persistence: object) -> None:
        del persistence
        closed["count"] += 1

    monkeypatch.setattr(
        host_runtime_module,
        "close_host_owned_diagnostic_persistence",
        _record_close,
    )
    reader = _RecordingReconstructionReader()
    runtime = _build_product_host(
        tmp_path,
        overrides=DiagnosticCompositionOverrides(execution_reconstruction_reader=reader),
    )
    _exercise_host_bound_consumers(runtime)
    close_harness_host_runtime(runtime)
    assert closed["count"] == 1
