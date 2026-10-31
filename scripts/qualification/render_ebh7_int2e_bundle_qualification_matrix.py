# © Artur Czarnecki. All rights reserved.

"""Generate INT2E bundle qualification matrix from git diff @ gap-closure."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_GAP_START = "ad3e24a20d7d1b341f721ef2427b9c1f65506061"
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_INT2E_BUNDLE_QUALIFICATION_MATRIX.md"


def _changed_bundle_modules() -> tuple[str, ...]:
    out = subprocess.check_output(
        ["git", "diff", f"{_GAP_START}..HEAD", "--name-only", "--", "intergrax/integrations/providers/"],
        cwd=_REPO,
        text=True,
    )
    modules: list[str] = []
    for line in out.splitlines():
        if line.strip().endswith("bundle.py"):
            rel = line.strip().replace("\\", "/")
            mod = "intergrax." + rel.removeprefix("intergrax/").removesuffix(".py").replace("/", ".")
            modules.append(mod)
    return tuple(sorted(set(modules)))


def render() -> str:
    mods = _changed_bundle_modules()
    lines = [
        "# EBH-7 — INT2E bundle qualification matrix",
        "",
        f"**Gap-closure delta:** `{_GAP_START}..HEAD` · **Bundles:** {len(mods)}",
        "",
        "| Module | Import smoke | Factory resolve | Negative | Live PROD-Q |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mod in mods:
        lines.append(
            f"| `{mod}` | `test_ebh7_int2e_bundle_imports` | category-specific | unregistered fail-closed | Path B |"
        )
    lines.append("")
    lines.append(
        "Regression gate: "
        "[`test_ebh_7_int2e_bundle_regression.py`]"
        "(../../../tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
