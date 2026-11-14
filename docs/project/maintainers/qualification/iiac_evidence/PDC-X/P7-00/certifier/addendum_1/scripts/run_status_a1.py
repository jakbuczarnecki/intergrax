import os, sys, subprocess, time
import pytest
import intergrax
FILES = [
 "tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py",
 "tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py",
 "tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py",
 "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py",
 "tests/qualification/ebh_5/test_ebh_5_rpl_f_integrity_gates.py",
 "tests/unit/governance/test_pdc_x_inventory.py",
 "tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py",
 "tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py",
]
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
print("# certifier finding-status re-run (PDC-P7-00 preflight addendum 1): 3 EBH-5 gate files + test_pdc_x_inventory + 2 COMPAT-X event-evolution gate files (F-35)")
print("# command: python .tmp/session/p7-00-cert/run_gates.py  (pytest.main(FILES+['-q','-rfE']))")
print("# checkout:", os.getcwd()); print("# HEAD:", sha); print("# PYTHONPATH:", os.environ.get("PYTHONPATH")); print("# intergrax from:", intergrax.__file__)
sys.stdout.flush(); t=time.time()
rc = pytest.main(FILES+["-q","-rf","--tb=line"])
print("# pytest exit code:", int(rc)); print("# elapsed_s: %.2f" % (time.time()-t)); sys.stdout.flush()
os._exit(int(rc))
