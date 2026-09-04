# © Artur Czarnecki. All rights reserved.

"""Reusable typed multiprocessing workers for S1 scale qualification."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.system.functional_diagnostics_scale.mongodb_backend import resolve_mongodb_uri
from tests.system.functional_diagnostics_scale.process_ipc import (
    ScaleWorkerCommand,
    ScaleWorkerPhase,
    ScaleWorkerResult,
    parse_worker_result,
)
from tests.system.functional_diagnostics_scale.profile import FunctionalDiagnosticsScaleProfile

_PROCESS_PROBE_MODULE = "tests.system.functional_diagnostics_scale.process_probe"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUBPROCESS_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, slots=True)
class ScaleWorkerBatchResult:
    phase: ScaleWorkerPhase
    results: tuple[ScaleWorkerResult, ...]
    all_exit_ok: bool


class ScaleWorker:
    """Launch typed scale worker subprocesses."""

    def __init__(
        self,
        *,
        work_dir: Path,
        collection_name: str,
        cursor_secret_hex: str,
        profile: FunctionalDiagnosticsScaleProfile,
    ) -> None:
        self._work_dir = work_dir
        self._collection_name = collection_name
        self._cursor_secret_hex = cursor_secret_hex
        self._profile = profile
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def run_phase(
        self,
        phase: ScaleWorkerPhase,
        *,
        worker_count: int,
    ) -> ScaleWorkerBatchResult:
        results: list[ScaleWorkerResult] = []
        all_exit_ok = True
        for worker_index in range(worker_count):
            command = ScaleWorkerCommand(
                phase=phase,
                collection_name=self._collection_name,
                cursor_secret_hex=self._cursor_secret_hex,
                page_size=self._profile.page_size,
                query_page_limit=self._profile.document_store_query_page_limit,
                worker_index=worker_index,
                worker_count=worker_count,
                seed=self._profile.seed,
                profile_name=self._profile.name.value,
            )
            completed = subprocess.run(
                self._argv(command),
                cwd=_REPO_ROOT,
                env=self._env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                all_exit_ok = False
                results.append(
                    ScaleWorkerResult(
                        schema_version=1,
                        pid=0,
                        phase=phase.value,
                        worker_index=worker_index,
                        written_count=0,
                        read_count=0,
                        append_latency_ms=(),
                        read_latency_ms=(),
                        conflicts=0,
                        errors=1,
                        exit_code=completed.returncode,
                        detail=(completed.stderr or "worker failed").strip(),
                    ),
                )
                continue
            parsed = parse_worker_result(completed.stdout.strip().splitlines()[-1])
            results.append(parsed)
            if parsed.exit_code != 0:
                all_exit_ok = False
        return ScaleWorkerBatchResult(
            phase=phase,
            results=tuple(results),
            all_exit_ok=all_exit_ok,
        )

    def _argv(self, command: ScaleWorkerCommand) -> list[str]:
        return [
            sys.executable,
            "-m",
            _PROCESS_PROBE_MODULE,
            command.phase.value,
            "--collection-name",
            command.collection_name,
            "--cursor-secret-hex",
            command.cursor_secret_hex,
            "--page-size",
            str(command.page_size),
            "--query-page-limit",
            str(command.query_page_limit),
            "--worker-index",
            str(command.worker_index),
            "--worker-count",
            str(command.worker_count),
            "--seed",
            str(command.seed),
            "--profile-name",
            command.profile_name,
        ]

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["INTERGRAX_MONGODB_URI"] = resolve_mongodb_uri()
        env["INTERGRAX_MONGODB_DATABASE"] = "intergrax_diag_scale_s1"
        env["INTERGRAX_MONGODB_COLLECTION"] = self._collection_name
        pythonpath = [
            str(_REPO_ROOT),
            str(_REPO_ROOT / "agents"),
            str(_REPO_ROOT / "applications"),
        ]
        if existing := env.get("PYTHONPATH", "").strip():
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env


__all__ = ["ScaleWorker", "ScaleWorkerBatchResult"]
