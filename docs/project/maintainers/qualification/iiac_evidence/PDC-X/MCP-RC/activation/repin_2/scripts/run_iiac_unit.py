import os, sys, subprocess, time
import pytest, intergrax
FILES = [
    "tests/unit/governance/test_iiac_activation_gate.py",
    "tests/unit/governance/test_iiac_g0_inventory.py",
    "tests/unit/governance/test_iiac_negative_guards.py",
]
print("# activation certifier re-pin 2: IIAC governance unit tests, one pytest process")
print("# HEAD:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print("# checkout:", os.getcwd(), "| PYTHONPATH:", os.environ.get("PYTHONPATH"), "| intergrax from:", intergrax.__file__)
sys.stdout.flush()
t = time.time()
rc = pytest.main(FILES + ["-q", "-rf", "--tb=short"])
print("# pytest exit code:", int(rc))
print("# elapsed_s: %.2f" % (time.time() - t))
sys.stdout.flush()
os._exit(int(rc))
