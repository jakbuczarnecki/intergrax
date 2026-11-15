import os, sys, subprocess, time
import pytest, intergrax
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
print("# governance-fast probe: validate_policy_registry.py, then pytest tests/unit/governance/ -q --tb=line (workflow steps of unit-tests.yml)")
print("# HEAD:", sha); print("# PYTHONPATH:", os.environ.get("PYTHONPATH")); print("# intergrax from:", intergrax.__file__); sys.stdout.flush()
r=subprocess.run([sys.executable,"scripts/governance/validate_policy_registry.py"],capture_output=True,text=True)
print("# validate_policy_registry exit:", r.returncode, (r.stdout+r.stderr).strip()[-300:]); sys.stdout.flush()
t=time.time(); rc=pytest.main(["tests/unit/governance/","-q","--tb=line","-rf"])
print("# pytest exit code:", int(rc)); print("# elapsed_s: %.2f" % (time.time()-t)); sys.stdout.flush(); os._exit(int(rc))
