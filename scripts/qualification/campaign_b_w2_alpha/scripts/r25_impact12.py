# ruff: noqa: E501
"""R25c - build_impact_map.py --since-sha for the 12 stage cert SHAs, one child at a time (X-A6-003 / A6-003).
Usage: python -I r25_impact12.py <outdir>   (cwd = scratch worktree at B; PYTHONPATH must make 'intergrax' import from cwd)
The 12 SHAs are the ones recorded in campaign_a/TOOL_RUN_LOG.md (A5-impact-* rows)."""

import os
import subprocess
import sys
from pathlib import Path

SHAS = [
    ("TENANT-X", "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558"),
    ("CONFIG-X", "84b8739b8b6e62c8e580e26ca3011f6aa46d457a"),
    ("COMPAT-X", "84b8739b8b6e62c8e580e26ca3011f6aa46d457a"),
    ("STATE-X", "bd54941d933069b8bfb2819bb819c1cbdbe71576"),
    ("TRACE-X", "fac9a43f3b128fde63541d2b56ebd8f4a751e8fb"),
    ("CTRL-X", "bd54941d933069b8bfb2819bb819c1cbdbe71576"),
    ("GOV-X2", "318a4d835534ef5c48575f4649551c957dacd8ea"),
    ("EBH-5", "33ecf19cc020b112d9e586c87b49e819202f5d6d"),
    ("DUP-X", "a7d83925b375b5f08eb0f175ced2c17fa026bba9"),
    ("EVOL-X", "7a4f279fd8c0986431941c77283d872ea5b0c1ba"),
    ("ENT-AUDIT-X", "5f178174e2462b038d544b61192a666d5b0ca79f"),
    ("MCP-SERVER-X", "919e34b8f9fbccfc62a16205d8647d6e6070ce45"),
]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
wt = Path.cwd()
env = dict(os.environ, PYTHONPATH=str(wt), PYTHONDONTWRITEBYTECODE="1")
exit0 = 0
bad = []
for name, sha in SHAS:
    r = subprocess.run(
        [
            sys.executable,
            "scripts/enterprise_audit/build_impact_map.py",
            "--since-sha",
            sha,
            "--output",
            str(out / f"impact_{name}.json"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(wt),
    )
    tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
    print(f"IMPACT {name} {sha[:12]} exit={r.returncode} :: {tail[0][:160]}", flush=True)
    if r.returncode == 0:
        exit0 += 1
    else:
        bad.append((name, r.returncode))
print(f"R25c_SUMMARY exit0={exit0} nonzero={len(bad)} {bad}", flush=True)
sys.exit(0 if not bad else 1)
