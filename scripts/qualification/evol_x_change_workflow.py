# © Artur Czarnecki. All rights reserved.

"""EVOL-X change workflow CLI A1..A8 (qualification tooling, passive).

Exit codes: 0 = all stages PASS/SKIPPED; 2 = at least one stage FAILED; 1 = tool/IO error.
Stage JSON is written to ``--out-dir`` (default ``.tmp/session/evol-x-full-plan/workflow``).
``--write-baseline`` records the current digests as the accepted baseline and exits 0 (operator action).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testing_support.change_safety.assess.triggers import build_baseline_manifest, compute_actual_digests
from testing_support.change_safety.contracts.verdicts import AcceptedBaselineManifest
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.intake.git_change_set import GitChangeSetPort, rev_parse_head
from testing_support.change_safety.workflow import run_change_workflow

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO / ".tmp" / "session" / "evol-x-full-plan" / "workflow"


def _write_stage_files(out_dir: Path, report) -> None:  # type: ignore[no-untyped-def]
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage in report.stages:
        body = {
            "stage_id": stage.stage_id,
            "name": stage.name,
            "status": stage.status,
            "detail": stage.detail,
            "payload": stage.payload,
        }
        (out_dir / f"{stage.stage_id}_{stage.name}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    summary = {
        "exit_code": report.exit_code,
        "failed_stages": list(report.failed_stages),
        "stages": [{"id": s.stage_id, "status": s.status, "detail": s.detail} for s in report.stages],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EVOL-X change workflow A1..A8")
    parser.add_argument("--base-sha", default=None)
    parser.add_argument("--head-sha", default=None)
    parser.add_argument("--baseline", type=Path, default=None, help="accepted baseline manifest JSON")
    parser.add_argument("--write-baseline", type=Path, default=None, help="write current digests as baseline and exit")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--authorize-surface", action="append", default=[], help="frozen-core surface id with signed record"
    )
    parser.add_argument(
        "--declared-shared", action="append", default=[], help="dotted module prefix of declared shared surface"
    )
    args = parser.parse_args(argv)
    try:
        head = args.head_sha or rev_parse_head()
        inventory = build_closed_world_inventory(source_commit_sha=head)
        if args.write_baseline is not None:
            manifest = build_baseline_manifest(accepted_at_sha=head, digests=compute_actual_digests(inventory))
            args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
            args.write_baseline.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            sys.stdout.write(f"baseline written: {args.write_baseline}\n")
            return 0
        if not args.base_sha:
            parser.error("--base-sha is required unless --write-baseline")
        change_set = GitChangeSetPort().load(base_sha=args.base_sha, head_sha=head)
        baseline = (
            AcceptedBaselineManifest.model_validate_json(args.baseline.read_text(encoding="utf-8"))
            if args.baseline is not None
            else None
        )
        sys.path.insert(0, str(_REPO))
        from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs

        report = run_change_workflow(
            change_set=change_set,
            inventory=inventory,
            selection_inputs_factory=build_selection_inputs,
            owner_rules=load_evol_x_data_document("owner_rules.v1.json"),
            exclusion_register=load_evol_x_data_document("exclusion_register.v1.json"),
            baseline=baseline,
            authorized_surfaces=frozenset(args.authorize_surface),
            declared_shared=frozenset(args.declared_shared),
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and exit non-zero
        sys.stderr.write(f"evol_x_change_workflow error: {type(exc).__name__}: {exc}\n")
        return 1
    _write_stage_files(args.out_dir, report)
    for stage in report.stages:
        sys.stdout.write(f"{stage.stage_id} {stage.name:<16} {stage.status:<8} {stage.detail}\n")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
