# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Core certification report models and writers (HEP Band 2ae · EVID-CORE-05)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.evidence.core_certification_spec import (
    CoreCertificationLevel,
    CoreCertificationMode,
    CoreCertificationSurface,
)
from intergrax.runtime.evidence.scenario_contracts import CoreScenarioResult, CoreScenarioStatus

CORE_CERTIFICATION_REPORT_SCHEMA_VERSION = "1.0.0"
DEFAULT_CORE_CERTIFICATION_OUTPUT_DIR = Path("build/evidence/core_certification")


class CoreCertificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CORE_CERTIFICATION_REPORT_SCHEMA_VERSION
    certification_level: CoreCertificationLevel
    mode: CoreCertificationMode = CoreCertificationMode.OPERATOR_LOCAL
    surface: CoreCertificationSurface = CoreCertificationSurface.CERTIFY_CORE
    passed: bool
    scenarios_passed: int
    scenarios_failed: int
    scenarios_skipped: int = 0
    scenarios_total: int
    scenario_results: list[CoreScenarioResult] = Field(default_factory=list)
    certification_run_id: str
    output_dir: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def summarize_core_scenario_results(
    results: list[CoreScenarioResult],
) -> tuple[int, int, int]:
    passed = sum(1 for item in results if item.status is CoreScenarioStatus.PASSED)
    failed = sum(1 for item in results if item.status is CoreScenarioStatus.FAILED)
    skipped = sum(1 for item in results if item.status is CoreScenarioStatus.SKIPPED)
    return passed, failed, skipped


def build_core_certification_report(
    *,
    level: CoreCertificationLevel,
    results: list[CoreScenarioResult],
    certification_run_id: str,
    output_dir: Path,
    mode: CoreCertificationMode = CoreCertificationMode.OPERATOR_LOCAL,
) -> CoreCertificationReport:
    passed_count, failed_count, skipped_count = summarize_core_scenario_results(results)
    total = len(results)
    all_passed = failed_count == 0 and skipped_count == 0 and passed_count == total
    return CoreCertificationReport(
        certification_level=level,
        mode=mode,
        passed=all_passed,
        scenarios_passed=passed_count,
        scenarios_failed=failed_count,
        scenarios_skipped=skipped_count,
        scenarios_total=total,
        scenario_results=list(results),
        certification_run_id=certification_run_id,
        output_dir=str(output_dir),
    )


def format_core_certification_markdown(report: CoreCertificationReport) -> str:
    lines = [
        "# Intergrax Core Certification Report",
        "",
        f"- **Level:** {report.certification_level.value}",
        f"- **Passed:** {report.passed}",
        f"- **Run ID:** {report.certification_run_id}",
        f"- **Generated:** {report.generated_at.isoformat()}",
        f"- **Scenarios:** {report.scenarios_passed}/{report.scenarios_total} passed",
        "",
        "## Scenario results",
        "",
        "| Scenario | Status | Message |",
        "| --- | --- | --- |",
    ]
    for result in report.scenario_results:
        message = result.message.replace("|", "\\|")
        lines.append(f"| {result.scenario_id} | {result.status.value} | {message} |")
    lines.append("")
    return "\n".join(lines)


def write_core_certification_report(
    report: CoreCertificationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_core_certification_markdown(report), encoding="utf-8")
    return json_path, md_path
