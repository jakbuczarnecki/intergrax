"""Session C Mode-C probe: tamper detection of P03 Mode A (consistent-digest adversary). Throwaway worktree only."""
import hashlib
import json
import subprocess
from pathlib import Path

from testing_support.change_safety.evidence import p03

WT = Path(".").resolve()
FINAL = "bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1"
B = WT / p03.BUNDLE_REL
NL = chr(10)


def git(*a):
    cmd = ["git", "-c", "user.name=iiac-c-probe", "-c", "user.email=probe@invalid", "-c", "commit.gpgsign=false", *a]
    return subprocess.run(cmd, cwd=WT, check=True, capture_output=True, text=True).stdout


def reset():
    git("checkout", "-f", "--detach", FINAL)


def rewrite_digests():
    """Re-sign ARTIFACT_DIGESTS over STAGED GIT BLOB bytes (avoids Windows CRLF skew); skip deleted files."""
    m = json.loads((B / "MANIFEST.json").read_text(encoding="utf-8"))
    git("add", "-A")
    digs = {}
    for rel in sorted(m["artifact_paths"]):
        r = subprocess.run(["git", "show", f":{p03.BUNDLE_REL}/{rel}"], cwd=WT, capture_output=True)
        if r.returncode == 0:
            digs[rel] = hashlib.sha256(r.stdout).hexdigest()
    old = json.loads((B / p03.DIGESTS_NAME).read_text(encoding="utf-8"))
    payload = {"digests": digs, "schema_version": old["schema_version"]}
    (B / p03.DIGESTS_NAME).write_text(json.dumps(payload, indent=2) + NL, encoding="utf-8", newline=NL)


def run(label, mutate, redigest):
    reset()
    mutate()
    if redigest:
        rewrite_digests()
    git("add", "-A")
    git("commit", "-m", f"probe {label}", "--allow-empty")
    problems = p03.verify_mode_a(WT)
    verdict = "DETECTED" if problems else "NOT_DETECTED(!)"
    print(f"[{verdict}] {label}: {problems[:3]}")


def edit_json(rel, fn):
    p = B / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    fn(d)
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + NL, encoding="utf-8", newline=NL)


def tamper_c2():
    p = B / "m09_automated_diagnostic/C2.json"
    p.write_text(p.read_text(encoding="utf-8").replace("module", "modulX", 1), encoding="utf-8", newline=NL)


reset()
print("CONTROL FINAL:", p03.verify_mode_a(WT) or "OK")
run("T1 byte tamper, digests NOT updated", lambda: edit_json("evol_metrics.v1.json", lambda d: d["readings"][0].update(numerator=1)), False)
run("T2 metric flipped to failing, digests re-signed", lambda: edit_json("evol_metrics.v1.json", lambda d: d["readings"][0].update(passed=False)), True)
run("T3 FRZ-07 scope footnote removed, re-signed", lambda: edit_json("evol_frz_evidence.v1.json", lambda d: [r.update(scope_footnote="") for r in d["rows"] if r["frz_id"] == "FRZ-EVOL-07"]), True)
run("T4 FRZ-04 row failed, re-signed", lambda: edit_json("evol_frz_evidence.v1.json", lambda d: d["rows"][3].update(passed=False)), True)
run("T5 experiment S6 negative arm failed, re-signed", lambda: edit_json("experiments/S6_negative.json", lambda d: d.update(passed=False)), True)
run("T6 M-09H forged MEASURED/PASS, re-signed", lambda: edit_json("evol_metrics.v1.json", lambda d: [r.update(status="MEASURED", passed=True, numerator=3, denominator=3) for r in d["readings"] if r["metric_id"] == "EVOL-M-09H"]), True)
run("T7 stale source_commit_sha in metrics, re-signed", lambda: edit_json("evol_metrics.v1.json", lambda d: d.update(source_commit_sha="0" * 40)), True)
run("T8 junit-outcomes: one test flipped to failed, re-signed", lambda: edit_json("evol_junit_outcomes.v1.json", lambda d: d["outcomes"].update({next(iter(d["outcomes"])): "failed"})), True)
run("T9 M-09A qual: human_m09h_measured forged true, re-signed", lambda: edit_json("evol_m09_automated_diagnostic_qualification.v1.json", lambda d: d.update(human_m09h_measured=True)), True)
run("T10 M-09A diagnostic output C2 byte tamper, re-signed", tamper_c2, True)
run("T11 delete experiment verdict S4_negative, re-signed", lambda: (B / "experiments/S4_negative.json").unlink(), True)
run("T12 manifest certification_profile flipped to HUMAN_FULL, re-signed", lambda: edit_json("MANIFEST.json", lambda d: d.update(certification_profile="HUMAN_FULL")), True)
reset()
