"""Run verify_iiac_audit_manifest and verify_iiac_role_separation on manifests, one process at a time.

Usage (from the checkout root): python run_manifest_validators.py <log-name> <manifest> [<manifest> ...]
Flags: audit manifest with --require-handoff-template; role separation with --strict-product-diff.
Writes ../logs/<log-name>.log (command, HEAD, UTC, exit code per call). Exit 0 only if every call exits 0.
"""
import datetime
import subprocess
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    name, manifests = sys.argv[1], sys.argv[2:]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = [f"# HEAD: {head}\n# python: {sys.executable}\n"]
    worst = 0
    for m in manifests:
        for mod, extra in (
            ("scripts.governance.verify_iiac_audit_manifest", ["--repo", ".", "--require-handoff-template"]),
            ("scripts.governance.verify_iiac_role_separation", ["--strict-product-diff", "--repo", "."]),
        ):
            cmd = [sys.executable, "-m", mod, m, *extra]
            t0 = utc()
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            t1 = utc()
            out.append(
                f"\n# command: {' '.join(cmd)}\n# UTC start: {t0}  end: {t1}\n{p.stdout}{p.stderr}# exit code: {p.returncode}\n"
            )
            worst = worst or p.returncode
    body = "".join(out) + f"\n# overall exit: {worst}\n"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{name}.log").write_text(body, encoding="utf-8", newline="\n")
    sys.stdout.write(body)
    return worst


if __name__ == "__main__":
    sys.exit(main())
