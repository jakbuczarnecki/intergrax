"""One-click VPI Data Pack resume launcher (preflight + subprocess; no fresh start)."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.scripts.dataset.operator_lifecycle.vpi_data_pack_resume_config import (
    VPI_DATA_PACK_PROCESS_JSON,
    VPI_DATA_PACK_RESUME_BUILD_SOURCE_SHA,
    VpiDataPackResumeLaunchPlan,
    build_vpi_data_pack_resume_cli_argv,
    resolve_vpi_data_pack_resume_launch_plan,
)

_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True, slots=True)
class VpiDataPackResumePreflightError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _windows_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    exit_code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
    kernel32.CloseHandle(handle)
    return int(exit_code.value) == _STILL_ACTIVE


def _read_process_pids(process_json: Path) -> tuple[int | None, int | None]:
    if not process_json.is_file():
        return None, None
    try:
        payload = json.loads(process_json.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise VpiDataPackResumePreflightError(
            f"invalid process evidence: {process_json}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise VpiDataPackResumePreflightError(
            f"invalid process evidence: {process_json}"
        ) from exc
    python_pid = payload.get("python_pid")
    powershell_pid = payload.get("powershell_pid")
    py = int(python_pid) if isinstance(python_pid, int) else None
    ps = int(powershell_pid) if isinstance(powershell_pid, int) else None
    return py, ps


def assert_no_active_canonical_writer(
    process_json: Path = VPI_DATA_PACK_PROCESS_JSON,
) -> None:
    python_pid, powershell_pid = _read_process_pids(process_json)
    for label, pid in (("python", python_pid), ("powershell", powershell_pid)):
        if pid is None:
            continue
        if _windows_pid_alive(pid):
            raise VpiDataPackResumePreflightError(
                "VPI DATA PACK IS ALREADY RUNNING\nNO SECOND WRITER STARTED "
                f"({label} pid={pid})"
            )


def assert_cuda_preflight(cuda_python: Path) -> str:
    if not cuda_python.is_file():
        raise VpiDataPackResumePreflightError(f"CUDA python missing: {cuda_python}")
    script = (
        "import torch\n"
        "assert torch.cuda.is_available()\n"
        "name = torch.cuda.get_device_name(0)\n"
        "print(torch.__version__)\n"
        "print(name)\n"
    )
    completed = subprocess.run(
        [str(cuda_python), "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise VpiDataPackResumePreflightError(
            f"CUDA preflight failed (exit {completed.returncode}): {stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    gpu_name = lines[-1] if lines else "unknown"
    return gpu_name


def assert_path_preflight(plan: VpiDataPackResumeLaunchPlan) -> None:
    missing: list[str] = []
    if not plan.cuda_python.is_file():
        missing.append(str(plan.cuda_python))
    if not plan.dataset_path.is_file():
        missing.append(str(plan.dataset_path))
    if not plan.dataset_manifest_path.is_file():
        missing.append(str(plan.dataset_manifest_path))
    if not plan.output_root.is_dir():
        missing.append(str(plan.output_root))
    if not plan.build_source_root.is_dir():
        missing.append(str(plan.build_source_root))
    paths = resolve_data_pack_paths(plan.output_root)
    if not paths.build_state_file.is_file():
        missing.append(str(paths.build_state_file))
    if missing:
        raise VpiDataPackResumePreflightError(
            "resume path preflight failed; missing:\n" + "\n".join(missing)
        )
    if VPI_DATA_PACK_RESUME_BUILD_SOURCE_SHA == "PENDING_R1_COMMIT":
        raise VpiDataPackResumePreflightError(
            "resume build-source SHA not pinned; create immutable snapshot first"
        )


def print_operator_banner(
    *,
    plan: VpiDataPackResumeLaunchPlan,
    ready_shards: int,
    total_shards: int,
    resume_shard: int | None,
    boundary_shard: int | None,
    gpu_name: str,
) -> None:
    resume_label = str(resume_shard) if resume_shard is not None else "complete"
    boundary_label = str(boundary_shard) if boundary_shard is not None else "none"
    print("=" * 60)
    print("VPI DATA PACK — RESUME")
    print("=" * 60)
    print(f"artifact: {plan.output_root}")
    print(f"ready shards: {ready_shards} / {total_shards}")
    print(f"resume shard: {resume_label}")
    print(f"boundary validation: shard {boundary_label} only")
    print(f"profile: {plan.execution_profile}")
    print(f"GPU: {gpu_name}")
    print("=" * 60)


def _read_build_progress_counts(output_root: Path) -> tuple[int, int, int | None, int | None]:
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resume_policy import (
        resolve_resume_boundary,
    )
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
        read_build_state_file,
    )

    paths = resolve_data_pack_paths(output_root)
    state = read_build_state_file(paths.build_state_file)
    boundary = resolve_resume_boundary(state)
    return (
        boundary.completed_shards,
        state.shard_count,
        boundary.next_shard_ordinal,
        boundary.last_ready_ordinal,
    )


def run_vpi_data_pack_resume(*, dry_run: bool = False) -> int:
    plan = resolve_vpi_data_pack_resume_launch_plan()
    try:
        assert_no_active_canonical_writer()
        gpu_name = assert_cuda_preflight(plan.cuda_python)
        assert_path_preflight(plan)
    except VpiDataPackResumePreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ready, total, resume_shard, boundary_shard = _read_build_progress_counts(
        plan.output_root
    )
    print_operator_banner(
        plan=plan,
        ready_shards=ready,
        total_shards=total,
        resume_shard=resume_shard,
        boundary_shard=boundary_shard,
        gpu_name=gpu_name,
    )
    argv = build_vpi_data_pack_resume_cli_argv(plan)
    if dry_run:
        print("dry-run:", " ".join(argv))
        return 0
    completed = subprocess.run(
        argv,
        cwd=str(plan.build_source_root),
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    dry_run = "--dry-run" in args
    return run_vpi_data_pack_resume(dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
