# © Artur Czarnecki. All rights reserved.

"""Deliberate generator: F10 computed matrix + ECAP §14 computed block (tests verify read-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from tests.qualification.ebh_7._ebh7_f10_canonical_inputs import (  # noqa: E402
    ECAP_REPORT_REL,
    F10_MATRIX_REL,
    build_canonical_f10_inputs,
)
from tests.qualification.ebh_7._ebh7_f10_ecap_normative_matrix import (  # noqa: E402
    REPORT_BLOCK_BEGIN,
    REPORT_BLOCK_END,
    certify_f10_for_render,
    render_f10_matrix_markdown,
    render_report_block,
)
from tests.qualification.ebh_7._ebh7_f10_types import EcapGate  # noqa: E402


def _replace_report_block(report: str, block: str) -> str:
    if REPORT_BLOCK_BEGIN not in report or REPORT_BLOCK_END not in report:
        raise SystemExit(f"{ECAP_REPORT_REL} lacks F10 computed block markers")
    head, rest = report.split(REPORT_BLOCK_BEGIN, 1)
    _, tail = rest.split(REPORT_BLOCK_END, 1)
    return f"{head}{REPORT_BLOCK_BEGIN}\n{block}\n{REPORT_BLOCK_END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if files differ instead of writing")
    args = parser.parse_args(argv)

    inputs = build_canonical_f10_inputs()
    upstream = certify_f10_for_render(inputs).verdicts
    upstream = tuple(v for v in upstream if v.gate is not EcapGate.C5)
    report_path = _REPO / ECAP_REPORT_REL
    report = _replace_report_block(
        report_path.read_text(encoding="utf-8"), render_report_block(upstream, inputs.test_execution_sha)
    )
    stale = report != inputs.report_text
    if not args.check:
        report_path.write_text(report, encoding="utf-8")
        inputs = build_canonical_f10_inputs()
    matrix = render_f10_matrix_markdown(certify_f10_for_render(inputs))
    stale = stale or matrix != inputs.matrix_disk_text
    if args.check:
        print("STALE" if stale else "IN SYNC")
        return 1 if stale else 0
    (_REPO / F10_MATRIX_REL).write_text(matrix, encoding="utf-8")
    cert = certify_f10_for_render(inputs)
    print(cert.disposition)
    for verdict in cert.verdicts:
        print(f"{verdict.gate.value} {verdict.verdict.value} n={verdict.obligations} failed={len(verdict.failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
