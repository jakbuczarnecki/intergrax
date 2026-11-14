import os, sys, subprocess, time
import pytest, intergrax
ARGS = ["tests/qualification/ent_audit_x/unit/test_task_id_registry.py",
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py",
        "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py","-v","-rf","--tb=line"]
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
print("# ent-audit targeted probe: pr-gate files + VBN-REACHABILITY entrypoint file + roadmap_replay_x mechanical gates (FAST manifest nodes)")
print("# HEAD:", sha); print("# intergrax from:", intergrax.__file__); sys.stdout.flush()
t=time.time(); rc=pytest.main(ARGS)
print("# pytest exit code:", int(rc)); print("# elapsed_s: %.2f" % (time.time()-t)); sys.stdout.flush(); os._exit(int(rc))
