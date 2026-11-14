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
 "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py",
 "tests/qualification/pdc_x/p6_ecapfix/test_p6_ecapfix_register_catalog_slug.py",
]
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
print("# certifier finding-status re-run (PDC-P7-00 preflight addendum 2): F-41 (2 files), F-06 (3 EBH-5 gate files), F-07 inventory, F-35 (2 COMPAT-X files), F-42 (ebh_5 package_c_v2) and the new p6_ecapfix tests")
print("# command: python .tmp/session/p7-00-cert/run_status_a2.py  (pytest.main(FILES+['-q','-rfE']))")
print("# checkout:", os.getcwd()); print("# HEAD:", sha); print("# PYTHONPATH:", os.environ.get("PYTHONPATH")); print("# intergrax from:", intergrax.__file__)
sys.stdout.flush(); t=time.time()
rc = pytest.main(FILES+["-q","-rf","--tb=line"])
print("# pytest exit code:", int(rc)); print("# elapsed_s: %.2f" % (time.time()-t)); sys.stdout.flush()
os._exit(int(rc))
