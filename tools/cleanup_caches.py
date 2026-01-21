# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


TARGET_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
}


def iter_target_dirs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir() and path.name in TARGET_DIR_NAMES:
            yield path


def main() -> None:
    root = Path(".").resolve()
    removed = []

    for d in iter_target_dirs(root):
        try:
            shutil.rmtree(d)
            removed.append(d)
        except Exception as exc:
            print(f"[WARN] Failed to remove {d}: {exc}")

    if removed:
        print("Removed cache directories:")
        for d in removed:
            print(f" - {d}")
    else:
        print("No cache directories found.")


if __name__ == "__main__":
    main()
