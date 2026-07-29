# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint for LKW local model qualification benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

_APPLICATIONS_ROOT = Path(__file__).resolve().parents[2]
if str(_APPLICATIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLICATIONS_ROOT))

from local_workspace_application.benchmarks.local_model_qualification.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
