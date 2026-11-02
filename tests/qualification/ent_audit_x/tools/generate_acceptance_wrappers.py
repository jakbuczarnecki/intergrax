# © Artur Czarnecki. All rights reserved.

"""Generate 85 acceptance wrapper modules under acceptance/."""

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

from pathlib import Path

from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_acceptance_{slug}_manifest_entrypoint_exists() -> None:
    checks = load_default_manifest()
    match = next(c for c in checks if c.check_id == "{check.check_id}")
    assert match.entrypoint_kind != "internal_pending"
    root = Path(__file__).resolve().parents[4]
    if "::" in match.entrypoint:
        rel = match.entrypoint.split("::", 1)[0]
    else:
        rel = match.entrypoint
    assert (root / rel).is_file(), rel
'''
        path.write_text(content, encoding="utf-8")
    init = ACCEPT / "__init__.py"
    if not init.is_file():
        init.write_text("", encoding="utf-8")
    print(f"wrote {len(checks)} acceptance wrappers")


if __name__ == "__main__":
    main()
