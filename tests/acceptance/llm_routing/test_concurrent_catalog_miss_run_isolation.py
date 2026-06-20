# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

import pytest

from intergrax.llm_adapters.registry.catalog_miss_diag import (
    CatalogResolutionTier,
    ModelCatalogMissDiagV1,
    begin_catalog_miss_run,
    bind_catalog_miss_run_observer,
    reset_catalog_miss_diagnostics,
)
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens


@pytest.mark.integration
@pytest.mark.gate
def test_parallel_catalog_miss_runs_do_not_cross_contaminate() -> None:
    reset_catalog_miss_diagnostics()
    received: dict[str, list[ModelCatalogMissDiagV1]] = {"run-a": [], "run-b": []}
    lock = threading.Lock()

    def _run(run_id: str) -> None:
        begin_catalog_miss_run(run_id)

        def _observer(diag: ModelCatalogMissDiagV1) -> None:
            with lock:
                received[run_id].append(diag)

        bind_catalog_miss_run_observer(run_id, _observer)
        resolve_context_window_tokens(
            "openrouter",
            f"vendor/isolated-{run_id}",
            profile_options={"run_id": run_id},
        )

    threads = [
        threading.Thread(target=_run, args=("run-a",)),
        threading.Thread(target=_run, args=("run-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(received["run-a"]) == 1
    assert len(received["run-b"]) == 1
    assert received["run-a"][0].model_id == "vendor/isolated-run-a"
    assert received["run-b"][0].model_id == "vendor/isolated-run-b"
    assert received["run-a"][0].resolution_tier == CatalogResolutionTier.PROVIDER_DEFAULT.value
    assert received["run-b"][0].run_id == "run-b"
