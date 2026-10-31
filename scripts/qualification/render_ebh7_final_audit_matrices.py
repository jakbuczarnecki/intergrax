# © Artur Czarnecki. All rights reserved.

"""Generate EBH-7 final independent audit matrices appendix from qualification SSOT."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT_MASTER = _REPO / "docs/project/maintainers/qualification/EBH_7_FINAL_INDEPENDENT_AUDIT_REMEDIATION_MASTER_PLAN.md"
_OUT_APPENDIX = _REPO / "docs/project/maintainers/qualification/EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md"
_GAP_START = "ad3e24a20d7d1b341f721ef2427b9c1f65506061"


def _head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def _production_diff_paths() -> tuple[str, ...]:
    out = subprocess.check_output(
        ["git", "diff", f"{_GAP_START}..HEAD", "--name-only", "--", "intergrax/", "agents/", "applications/"],
        cwd=_REPO,
        text=True,
    )
    return tuple(line.strip() for line in out.splitlines() if line.strip())


def _render_rpl_section() -> list[str]:
    from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_row

    lines = [
        "## RPL-H matrix (56 rows)",
        "",
        "| CONCERN_ID | CLAIMED_EXTENSIBILITY | RPL_A..G | CURRENT_PROOF | "
        "VERIFIED_STATUS | DEFECT | TASK | ACCEPTANCE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for matrix_row in CROSS_LAYER_REPLACEABILITY_MATRIX:
        row = normative_rpl_row(matrix_row.concern_id)
        rpl_gates = "/".join(
            v.value
            for v in (
                matrix_row.rpl_a,
                matrix_row.rpl_b,
                matrix_row.rpl_c,
                matrix_row.rpl_d,
                matrix_row.rpl_e,
                matrix_row.rpl_f,
                matrix_row.rpl_g,
            )
        )
        lines.append(
            f"| {row.concern_id} | {matrix_row.rpl_h.value} | {rpl_gates} | {row.positive_test} | "
            f"{row.verified_status.value} | {row.defect} | {row.task_id} | {row.acceptance} |"
        )
    lines.append("")
    return lines


def _render_ten_section() -> list[str]:
    from tests.qualification.ebh_7._ebh7_scenario_normative_audit import normative_ten_row
    from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

    lines = [
        "## TEN E2E matrix (26 rows)",
        "",
        "| ID | NORMATIVE_INVARIANT | ENTRYPOINT | TEST_NODE | VERIFIED_STATUS | REMEDIATION | EXIT |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in EBH7_TENANT_E2E_SCENARIOS:
        row = normative_ten_row(scenario)
        proof = f"{scenario.proof_module}::{scenario.proof_function}" if scenario.proof_module else "MISSING"
        lines.append(
            f"| {scenario.scenario_id} | {scenario.production_path} | {scenario.entrypoint} | {proof} | "
            f"{row.verified_status.value} | {row.remediation} | {row.exit_criterion} |"
        )
    lines.append("")
    return lines


def _render_xdm_section() -> list[str]:
    from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
    from tests.qualification.ebh_7._ebh7_scenario_normative_audit import normative_xdm_row

    lines = [
        "## XDM matrix (34 rows)",
        "",
        "| ID | COMBO_INVARIANT | COMPONENTS | TEST_NODE | VERIFIED_STATUS | REMEDIATION | EXIT |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in EBH7_CROSS_DOMAIN_SCENARIOS:
        row = normative_xdm_row(scenario)
        proof = f"{scenario.proof_module}::{scenario.proof_function}" if scenario.proof_module else "MISSING"
        lines.append(
            f"| {scenario.scenario_id} | {scenario.combo} | {scenario.components} | {proof} | "
            f"{row.verified_status.value} | {row.remediation} | {row.exit_criterion} |"
        )
    lines.append("")
    return lines


def _render_production_regression_section(paths: tuple[str, ...]) -> list[str]:
    lines = [
        "## Production change regression matrix (gap-closure range)",
        "",
        f"Range: `{_GAP_START}`..HEAD · **{len(paths)}** paths",
        "",
        "| PATH | REGRESSION_TASK |",
        "| --- | --- |",
    ]
    for path in paths:
        if "integrations/providers" in path and path.endswith("bundle.py"):
            task = "EBH7-W1-INT2E-BUNDLE"
        else:
            task = "EBH7-W1-PROD-DELTA"
        lines.append(f"| `{path}` | {task} |")
    lines.append("")
    return lines


def render_appendix_markdown() -> str:
    head = _head_sha()
    paths = _production_diff_paths()
    lines = [
        "# EBH-7 — Final audit matrices appendix (generated SSOT mirror)",
        "",
        f"**Generated @ HEAD:** `{head}`",
        "",
        "Regenerate: `uv run python scripts/qualification/render_ebh7_final_audit_matrices.py`",
        "",
        *_render_rpl_section(),
        *_render_ten_section(),
        *_render_xdm_section(),
        *_render_production_regression_section(paths),
    ]
    return "\n".join(lines)


def render_master_plan_stub() -> str:
    head = _head_sha()
    return f"""# EBH-7 — Final Independent Audit Remediation Master Plan (tracked hub)

**Program HEAD @ publication:** `{head}`

Canonical Cursor plan content is mirrored here for auditors. **Do not edit** `.cursor/plans/` copies.

## Execution waves

See generated appendix for per-row TASK ids. Waves 0–8: baseline, prod regression, RPL truth,
TEN/XDM, ECAP, L3, Protocol 03 B1–B5, B6, OD-08.

## Non-claims

B6 PASS, OD-EBH7-08 APPROVED, EBH-7 parent CLOSED, EVOL-X entry, global FRZ PASS —
operator/external only until Wave 7–8 complete.

## Appendix

[`EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md`](EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md)
"""


def main() -> None:
    appendix = render_appendix_markdown()
    _OUT_APPENDIX.write_text(appendix, encoding="utf-8")
    if not _OUT_MASTER.is_file():
        _OUT_MASTER.write_text(render_master_plan_stub(), encoding="utf-8")
    print(f"Wrote {_OUT_APPENDIX.relative_to(_REPO)} ({len(appendix)} bytes)")


if __name__ == "__main__":
    main()
