"""NT-RW-GATES-PROMO-DENY — unrelated governance ticket must not authorize promotion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_SCRIPT = REPO_ROOT / "scripts" / "gates" / "check_mvp_promotion_gates.py"


def _gate():
    spec = importlib.util.spec_from_file_location("rw_gates_mvp_gate", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ticket(tmp_path: Path, name: str, body: str) -> str:
    folder = tmp_path / "governance" / "tickets"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(body, encoding="utf-8")
    return f"governance/tickets/{name}"


def test_unrelated_ticket_denied_when_promotion_binding_requires_match(tmp_path: Path) -> None:
    gate = _gate()
    approved_f = _ticket(
        tmp_path,
        "T-IGX-051-F.md",
        "# F\n\n## Approved\n\nApproved: Operator @ 2026-11-20\n",
    )
    approved_other = _ticket(
        tmp_path,
        "T-IGX-999-Z.md",
        "# Z\n\n## Approved\n\nApproved: Operator @ 2026-11-20\n",
    )
    ok, detail = gate.promotion_ticket_authorization(approved_other, approved_f, tmp_path)
    assert ok is False
    assert "does not match promotion ticket" in detail


def test_matching_promotion_ticket_authorizes_when_approved(tmp_path: Path) -> None:
    gate = _gate()
    ticket = _ticket(tmp_path, "T-IGX-051-F.md", "# F\n\n## Approved\n\nApproved: Operator @ 2026-11-20\n")
    ok, _detail = gate.promotion_ticket_authorization(ticket, ticket, tmp_path)
    assert ok is True
