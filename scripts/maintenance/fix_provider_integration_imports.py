#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Fix typing imports after provider legacy delegate migration."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS_ROOT = ROOT / "intergrax" / "integrations" / "providers"


def fix_file(path: Path) -> bool:
    if "observability_backend" in path.as_posix():
        return False
    src = path.read_text(encoding="utf-8")
    if "@runtime_checkable" in src:
        return False
    original = src

    src = re.sub(
        r"from typing import Protocol, runtime_checkable, ([^\n]+), runtime_checkable",
        r"from typing import \1",
        src,
    )
    src = re.sub(
        r"from typing import Protocol, runtime_checkable, runtime_checkable",
        "",
        src,
    )
    src = re.sub(
        r"from typing import Protocol, runtime_checkable\n",
        "",
        src,
    )

    header, _, body = src.partition("class ")
    needs_any = bool(re.search(r"\bAny\b", body))
    has_any = "from typing import Any" in header or re.search(r"from typing import [^\n]*\bAny\b", header)

    if needs_any and not has_any:
        if "from typing import" in header:
            header = re.sub(
                r"(from typing import [^\n]+)\n",
                lambda m: m.group(1) + ", Any\n" if "Any" not in m.group(1) else m.group(0),
                header,
                count=1,
            )
        else:
            header = header.replace(
                "from __future__ import annotations\n\n",
                "from __future__ import annotations\n\nfrom typing import Any\n\n",
                1,
            )

    src = header + "class " + body
    src = re.sub(r"\n{4,}", "\n\n\n", src)

    if src != original:
        path.write_text(src, encoding="utf-8")
        return True
    return False


def main() -> None:
    fixed = 0
    for path in sorted(PROVIDERS_ROOT.glob("*/*/integration.py")):
        if fix_file(path):
            fixed += 1
    print(f"fixed imports: {fixed}")


if __name__ == "__main__":
    main()
