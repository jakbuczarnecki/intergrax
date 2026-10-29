# © Artur Czarnecki. All rights reserved.

"""Mandatory L9 blocker pytest nodeids — SSOT aligned with p03-handoff l9_blocker_replay."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.ebh_6._p03_handoff_export import export_l9_blocker_replay_bundle

_L9_BLOCKER_IDS: Final[frozenset[str]] = frozenset(
    {
        "EBH6-BLK-L9-OWN-01",
        "EBH6-BLK-L9-EXE-01",
        "EBH6-BLK-L9-GOV-01",
        "EBH6-BLK-L9-REC-01",
        "EBH6-BLK-L9-SEC-01",
        "EBH6-BLK-L9-REL-01",
        "EBH6-BLK-L9-HRN-01",
        "EBH6-BLK-L9-TEN-01",
    },
)

_PASS_LINE = re.compile(r"^(?P<node>.+?) PASSED\s+\[")


@lru_cache(maxsize=1)
def mandatory_l9_gate_nodeids() -> tuple[str, ...]:
    replay = export_l9_blocker_replay_bundle()
    assert set(replay) == set(_L9_BLOCKER_IDS)
    nodeids: list[str] = []
    for blocker_id in sorted(_L9_BLOCKER_IDS):
        gates = replay[blocker_id]["gate_test_ids"]
        nodeids.extend(str(g) for g in gates)
    unique = tuple(dict.fromkeys(nodeids))
    return unique


def parse_pytest_transcript_passed_nodeids(transcript: str) -> frozenset[str]:
    passed: set[str] = set()
    for line in transcript.splitlines():
        match = _PASS_LINE.match(line.strip())
        if match:
            passed.add(match.group("node"))
    return frozenset(passed)


def attest_l9_transcript(transcript_path: Path) -> dict[str, object]:
    mandatory = mandatory_l9_gate_nodeids()
    text = transcript_path.read_text(encoding="utf-8")
    passed = parse_pytest_transcript_passed_nodeids(text)
    rows: list[dict[str, str]] = []
    for node in mandatory:
        rows.append(
            {
                "nodeid": node,
                "status": "PASS" if node in passed else "MISSING_OR_FAIL",
            },
        )
    all_pass = all(row["status"] == "PASS" for row in rows)
    return {
        "mandatory_unique_nodeid_count": len(mandatory),
        "mandatory_nodeids": list(mandatory),
        "results": rows,
        "all_mandatory_pass": all_pass,
    }
