# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint for the governed hybrid knowledge flagship proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS = _REPO_ROOT / "applications"
for path in (_REPO_ROOT, _APPLICATIONS):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from proof_infrastructure.governed_hybrid_knowledge_proof.runner import run_flagship_proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the COMM-5D governed hybrid knowledge flagship proof "
            "(ORION deployment readiness, four scenarios)."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured proof result JSON after terminal output.",
    )
    args = parser.parse_args(argv)
    result = run_flagship_proof(emit_terminal=True)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
