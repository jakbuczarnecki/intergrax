import os, sys, subprocess, time
import pytest, intergrax
FILES = [
 "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py",
 "tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py",
 "tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py",
 "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py",
 "tests/unit/governance/test_pdc_x_inventory.py",
 "tests/unit/governance/test_iiac_activation_gate.py",
 "tests/unit/governance/test_iiac_g0_inventory.py",
 "tests/unit/governance/test_iiac_negative_guards.py",
]
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
print("# certifier finding-status re-run (PDC-P7-00 preflight addendum 3): F-56 (roadmap_replay_x closure counter), F-35 (COMPAT-X), F-42 (ebh_5 package_c_v2), F-07 inventory, IIAC governance unit tests (protocol edit)")
print("# HEAD:", sha, "| checkout:", os.getcwd(), "| PYTHONPATH:", os.environ.get("PYTHONPATH"), "| intergrax from:", intergrax.__file__); sys.stdout.flush(); t=time.time()
rc = pytest.main(FILES+["-q","-rf","--tb=line"])
print("# pytest exit code:", int(rc)); print("# elapsed_s: %.2f" % (time.time()-t)); sys.stdout.flush()
os._exit(int(rc))
