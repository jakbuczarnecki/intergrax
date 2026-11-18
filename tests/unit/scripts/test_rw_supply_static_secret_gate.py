# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part H: the static secret-literal gate (NT-RW-SUP-16) and the sandbox isolation-strength
gate (NT-RW-SUP-18).

Seeded sources are assembled from parts at run time so that this file holds no credential literal;
no assertion or output echoes a candidate value.
"""

from __future__ import annotations

import importlib.util
import secrets
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]


def _load_script(name: str) -> ModuleType:
    path = _REPO / "scripts" / "maintenance" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rw_supply_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def secret_gate() -> ModuleType:
    return _load_script("check_static_secret_literals")


@pytest.fixture(scope="module")
def sandbox_gate() -> ModuleType:
    return _load_script("check_sandbox_isolation_strength")


def _prefix() -> str:
    return "".join(("s", "k", "-"))


def _tail() -> str:
    return secrets.token_hex(16)


def _seeded_sources() -> dict[str, str]:
    """One module per escape shape; every value is generated at run time."""
    prefix, tail = _prefix(), _tail()
    return {
        "concatenation": f'API_KEY_VALUE = "{prefix}" + "{tail}"\n',
        "fstring": f'value = f"{prefix}{{"{tail}"}}"\n',
        "bytes": f'raw = b"{prefix}{tail}"\n',
        "joined_prefix": f'value = "".join(["{prefix[:1]}", "{prefix[1:]}", "{tail}"])\n',
        "or_fallback": f'import os\napi_key = os.environ.get("ACME_PROVIDER") or "{tail}"\n',
        "if_not_fallback": (
            f'def resolve(api_key):\n    if not api_key:\n        api_key = "{tail}"\n    return api_key\n'
        ),
    }


# --------------------------------------------------------------------- NT-RW-SUP-16 (F-D6-14 / PDC-D6-R007)


def test_sup_16a_seeded_evasions_flagged(secret_gate: ModuleType) -> None:
    expected_rules = {
        "concatenation": secret_gate.RULE_FOLDED_CONCATENATION,
        "fstring": secret_gate.RULE_CONSTANT_FSTRING,
        "bytes": secret_gate.RULE_BYTES_LITERAL,
        "joined_prefix": secret_gate.RULE_JOINED_PREFIX,
        "or_fallback": secret_gate.RULE_OR_FALLBACK,
        "if_not_fallback": secret_gate.RULE_IF_NOT_FALLBACK,
    }
    for shape, source in _seeded_sources().items():
        findings = secret_gate.scan_source_for_evasions(source, path=f"intergrax/seeded_{shape}.py")
        assert expected_rules[shape] in {finding.rule for finding in findings}, shape


def test_sup_16b_secret_literal_fails_gate_without_echo(
    secret_gate: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prefix, tail = _prefix(), _tail()
    module = tmp_path / "intergrax" / "leaky.py"
    module.parent.mkdir(parents=True)
    module.write_text(f'TOKEN_PART = "{prefix}" + "{tail}"\n', encoding="utf-8")
    assert secret_gate.main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "intergrax/leaky.py" in captured.out
    echoed = tail in captured.out + captured.err
    assert not echoed, "the gate output echoes the candidate value (value not shown)"

    module.write_text('LABEL = "plain-label"\n', encoding="utf-8")
    assert secret_gate.main(["--root", str(tmp_path)]) == 0


def test_sup_16c_product_tree_scan_recorded(secret_gate: ModuleType) -> None:
    """The real product tree is scanned and the finding count is recorded (no exit-0 assertion)."""
    findings = secret_gate.scan_product_tree(_REPO)
    rules = {finding.rule for finding in findings}
    known = {
        "env_secret_literal_fallback",
        "credential_format_literal",
        "secret_named_literal",
        "known_forbidden_literal",
        secret_gate.RULE_FOLDED_CONCATENATION,
        secret_gate.RULE_CONSTANT_FSTRING,
        secret_gate.RULE_BYTES_LITERAL,
        secret_gate.RULE_JOINED_PREFIX,
        secret_gate.RULE_OR_FALLBACK,
        secret_gate.RULE_IF_NOT_FALLBACK,
    }
    assert rules <= known
    print(f"static secret gate: {len(findings)} finding(s) over the product tree")


# --------------------------------------------------------------------- NT-RW-SUP-18 (PDC-D6-R010)


def test_sup_18a_weak_sandbox_refused(sandbox_gate: ModuleType) -> None:
    env = sandbox_gate.product_exec_profile()
    issues = sandbox_gate.isolation_strength_issues(env, sandbox_gate.weak_capabilities())
    assert "isolation_tier_too_weak" in issues
    assert "network_egress_not_enforced" in issues
    assert sandbox_gate.isolation_strength_issues(env, None) == ("sandbox_capabilities_unattested",)


def test_sup_18b_strong_sandbox_accepted(sandbox_gate: ModuleType) -> None:
    env = sandbox_gate.product_exec_profile()
    assert sandbox_gate.isolation_strength_issues(env, sandbox_gate.strong_capabilities()) == ()
    assert sandbox_gate.main() == 0
