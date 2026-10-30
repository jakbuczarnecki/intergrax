# © Artur Czarnecki. All rights reserved.

"""Scoped prod-Δ regression evidence chain validation (B6 R2-03)."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REPO = Path(__file__).resolve().parents[3]
_HISTORICAL_L2_TEST_EXECUTION_SHA: Final[str] = "22fbc0e837e819fe2a068295df5f527612dcf307"
_IMMUTABLE_PROOF_DIR: Final[str] = ".tmp/proof/roadmap-replay-x-immutable-6fbc4a8"
_EVIDENCE_RECORD_REL: Final[str] = (
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_QUALIFICATION_EVIDENCE_RECORD.md"
)
_APPROVED_DRIFT_LOG_REL: Final[str] = "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md"

_LOG_DIGEST_BY_NAME: Final[dict[str, str]] = {
    "l2-ctrl_x.log": "c9567cffca64f29fee608ddf01c63d7e7bd8c53eb0e18dac7936fd9eaa19413a",
}


@dataclass(frozen=True, slots=True)
class ScopedRegressionEvidence:
    prod_path: str
    historical_invariant: str
    regression_test_module: str
    regression_test_name: str
    batch_command: str
    test_execution_sha: str
    log_rel_path: str
    approved_evidence_ref: str
    batch_exit_code: int = 0


REPLAY_X_SCOPED_REGRESSION_EVIDENCE: Final[tuple[ScopedRegressionEvidence, ...]] = (
    ScopedRegressionEvidence(
        prod_path="intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
        historical_invariant="CTRL-X admission/shutdown semantics unchanged; pyright callback typing only",
        regression_test_module="tests/qualification/control_planes/ctrl_x/test_ctrl_x_typing_gates.py",
        regression_test_name="test_ctrl_x_semantic_boundary_pyright_zero_errors",
        batch_command="uv run pytest tests/qualification/control_planes/ctrl_x/ -q",
        test_execution_sha=_HISTORICAL_L2_TEST_EXECUTION_SHA,
        log_rel_path=f"{_IMMUTABLE_PROOF_DIR}/l2-ctrl_x.log",
        approved_evidence_ref=_APPROVED_DRIFT_LOG_REL,
    ),
)


def _test_function_exists(module_path: Path, test_name: str) -> bool:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            return True
    return False


def _read_log_text(log_path: Path) -> str:
    raw = log_path.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "passed" in text.lower() or "exit=0" in text.lower():
            return text
    return raw.decode("utf-8", errors="replace")


def _approved_ref_documents_path(repo: Path, evidence: ScopedRegressionEvidence) -> bool:
    ref_path = repo / evidence.approved_evidence_ref
    if not ref_path.is_file():
        return False
    text = ref_path.read_text(encoding="utf-8")
    return evidence.prod_path in text and evidence.regression_test_module in text


def scoped_regression_evidence_violations(repo: Path | None = None) -> tuple[str, ...]:
    root = _REPO if repo is None else repo
    violations: list[str] = []
    evidence_record = root / _EVIDENCE_RECORD_REL

    for row in REPLAY_X_SCOPED_REGRESSION_EVIDENCE:
        module = root / row.regression_test_module
        if not module.is_file():
            violations.append(f"missing_regression_module:{row.prod_path}")
            continue
        if not _test_function_exists(module, row.regression_test_name):
            violations.append(f"missing_regression_test_name:{row.regression_test_name}")

        if not _approved_ref_documents_path(root, row):
            violations.append(f"missing_approved_evidence_ref:{row.approved_evidence_ref}")

        if evidence_record.is_file():
            rec_text = evidence_record.read_text(encoding="utf-8")
            if row.test_execution_sha not in rec_text:
                violations.append(f"test_execution_sha_not_in_evidence_record:{row.test_execution_sha}")
            log_name = Path(row.log_rel_path).name
            if log_name not in rec_text:
                violations.append(f"log_not_in_evidence_record:{log_name}")
            if row.batch_command.split("tests/qualification/")[1].split()[0] not in rec_text.replace("`", ""):
                if "control_planes/ctrl_x" not in rec_text:
                    violations.append("batch_scope_not_in_evidence_record:ctrl_x")

        log_path = root / row.log_rel_path
        expected_digest = _LOG_DIGEST_BY_NAME.get(log_path.name)
        has_approved_chain = _approved_ref_documents_path(root, row) and evidence_record.is_file()
        if log_path.is_file():
            digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
            if expected_digest and digest != expected_digest:
                violations.append(f"log_digest_mismatch:{log_path.name}")
            log_text = _read_log_text(log_path).lower()
            if "passed" not in log_text and "exit=0" not in log_text:
                violations.append(f"log_missing_pass_evidence:{log_path.name}")
        elif not has_approved_chain:
            violations.append(f"missing_regression_log_or_approved_ref:{row.log_rel_path}")

    return tuple(violations)
