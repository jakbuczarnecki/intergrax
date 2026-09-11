"""Frozen production parameters for VPI 5C4F one-click resume (operator contract)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VPI_5C4F_ORIGINAL_BUILD_SOURCE_SHA = "1c8713ff60abdd5c9397182a01fc04d9e986354d"

# Immutable build-source snapshot for fast-resume R1 (updated when snapshot is created).
VPI_5C4F_RESUME_BUILD_SOURCE_SHA = "b0103d133528737484750fabd489ffbed964fedf"

VPI_5C4F_ARTIFACT_ROOT = Path(
    r"D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1"
)
VPI_5C4F_CUDA_PYTHON = Path(
    r"D:\Projekty\intergrax\.tmp\session\vpi-5c4a2\cuda-venv\Scripts\python.exe"
)
VPI_5C4F_DATASET_PATH = Path(
    r"D:\Projekty\intergrax\platform_proofs\scenarios\verified_product_identification"
    r"\dataset\processed\selected_offers.parquet"
)
VPI_5C4F_DATASET_MANIFEST_PATH = Path(
    r"D:\Projekty\intergrax\platform_proofs\scenarios\verified_product_identification"
    r"\dataset\processed\selected_offers_manifest.json"
)
VPI_5C4F_SHARD_SIZE = 1000
VPI_5C4F_EXECUTION_PROFILE = "production-local-gpu"
VPI_5C4F_OPERATOR_EVIDENCE_DIR = Path(
    r"D:\Projekty\intergrax-runtime-artifacts\vpi\operator-evidence\5c4f"
)
VPI_5C4F_PROCESS_JSON = VPI_5C4F_OPERATOR_EVIDENCE_DIR / "process.json"
VPI_5C4F_BUILD_SOURCE_ROOT = Path(
    r"D:\Projekty\intergrax-runtime-artifacts\vpi\build-source"
)


@dataclass(frozen=True, slots=True)
class Vpi5C4FResumeLaunchPlan:
    cuda_python: Path
    build_source_root: Path
    dataset_path: Path
    dataset_manifest_path: Path
    output_root: Path
    shard_size: int
    execution_profile: str
    resume: bool


def resolve_vpi_5c4f_resume_launch_plan() -> Vpi5C4FResumeLaunchPlan:
    build_source = VPI_5C4F_BUILD_SOURCE_ROOT / VPI_5C4F_RESUME_BUILD_SOURCE_SHA
    return Vpi5C4FResumeLaunchPlan(
        cuda_python=VPI_5C4F_CUDA_PYTHON,
        build_source_root=build_source,
        dataset_path=VPI_5C4F_DATASET_PATH,
        dataset_manifest_path=VPI_5C4F_DATASET_MANIFEST_PATH,
        output_root=VPI_5C4F_ARTIFACT_ROOT,
        shard_size=VPI_5C4F_SHARD_SIZE,
        execution_profile=VPI_5C4F_EXECUTION_PROFILE,
        resume=True,
    )


def build_resume_cli_argv(plan: Vpi5C4FResumeLaunchPlan) -> tuple[str, ...]:
    return (
        str(plan.cuda_python),
        "-m",
        "platform_proofs.scenarios.verified_product_identification.dataset.run_data_pack_build",
        "--dataset-path",
        str(plan.dataset_path),
        "--dataset-manifest-path",
        str(plan.dataset_manifest_path),
        "--output-root",
        str(plan.output_root),
        "--shard-size",
        str(plan.shard_size),
        "--resume",
        "--execution-profile",
        plan.execution_profile,
    )
