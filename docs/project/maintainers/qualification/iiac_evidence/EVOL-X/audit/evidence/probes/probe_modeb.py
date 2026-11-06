import json, sys
from pathlib import Path
from testing_support.change_safety.evidence import p03
from scripts.qualification.evol_x_p03_evidence import outcomes_from_junit
FINAL="bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1"
repo=Path(".").resolve()
base=outcomes_from_junit(Path(sys.argv[1]), repo)
print("replay outcomes:", len(base), set(base.values()))
def chk(label, o):
    try:
        pr=p03.verify_mode_b(repo, evidence_ref=FINAL, replay_outcomes=o, certification=True)
    except Exception as exc:
        print("[DETECTED_BY_UNHANDLED_EXCEPTION] "+label+": "+type(exc).__name__+" "+str(exc)[:120]); return
    print(("[DETECTED] " if pr else "[NOT_DETECTED(!)] ")+label+": "+str(pr[:2])[:300])
print("CONTROL:", p03.verify_mode_b(repo, evidence_ref=FINAL, replay_outcomes=base, certification=True) or "OK")
k=sorted(base)[0]
o=dict(base); o[k]="failed"; chk("MB1 one test failed", o)
o=dict(base); o[k]="skipped"; chk("MB2 one test skipped", o)
o=dict(base); del o[k]; chk("MB3 one node missing from replay", o)
o=dict(base); o["tests/qualification/evol_x/forged.py::test_x"]="passed"; chk("MB4 extra forged node", o)
chk("MB5 empty replay", {})
