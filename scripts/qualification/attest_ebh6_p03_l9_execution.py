# © Artur Czarnecki. All rights reserved.

"""Parse L9 pytest transcript and write attestation JSON (no pytest re-run)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/ebh_6_p03_independent_replay"
_TRANSCRIPT = _OUT / "l9-pytest-transcript.txt"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    return result.stdout.strip()


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from tests.qualification.ebh_6._l9_blocker_execution import attest_l9_transcript

    if not _TRANSCRIPT.is_file():
        print(f"Missing {_TRANSCRIPT}", file=sys.stderr)
        return 1
    payload = attest_l9_transcript(_TRANSCRIPT)
    payload["TEST_EXECUTION_SHA"] = _git_sha()
    payload["transcript_artifact"] = _TRANSCRIPT.name
    (_OUT / "l9-nodeid-attestation.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"all_mandatory_pass": payload["all_mandatory_pass"], "sha": payload["TEST_EXECUTION_SHA"]}))
    return 0 if payload["all_mandatory_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
