"""Generate 85 acceptance modules with semantic subprocess oracles (C033–C036)."""

from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.manifest import load_default_manifest

ROOT = Path(__file__).resolve().parents[1]
ACCEPT = ROOT / "acceptance"


def _slug(check_id: str) -> str:
    return check_id.lower().replace("-", "_")


def main() -> None:
    ACCEPT.mkdir(parents=True, exist_ok=True)
    checks = load_default_manifest()
    for check in checks:
        slug = _slug(check.check_id)
        path = ACCEPT / f"test_ent_a_{slug}.py"
        content = f'''from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.qualification.ent_audit_x.acceptance._gate_oracle import assert_semantic_gate_executes


@pytest.mark.ent_semantic_gate
def test_acceptance_{slug}_semantic_gate(repo_root: Path) -> None:
    if os.environ.get("ENT_AUDIT_X_SEMANTIC") != "1":
        pytest.skip("semantic gate matrix requires ENT_AUDIT_X_SEMANTIC=1")
    assert_semantic_gate_executes(repo_root, "{check.check_id}")
'''
        path.write_text(content, encoding="utf-8")
    init = ACCEPT / "__init__.py"
    if not init.is_file():
        init.write_text("", encoding="utf-8")
    print(f"wrote {len(checks)} acceptance semantic wrappers")


if __name__ == "__main__":
    main()
