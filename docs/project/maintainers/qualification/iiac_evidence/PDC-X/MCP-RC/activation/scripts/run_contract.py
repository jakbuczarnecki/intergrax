import os, sys, subprocess, time
import pytest
import intergrax
from tests.qualification.mcp_server_x.rc_selection import MCP_SERVER_X_RC_SELECTION as SEL
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("# activation certifier MCP-RC contract run (list passed from Python, one pytest process)")
print("# command: python .tmp/session/mcp-rc-act-cert/run_contract.py  (pytest.main(list(SELECTION)+['-q']))")
print("# checkout:", os.getcwd())
print("# HEAD:", sha)
print("# UTC start:", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
print("# PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("# intergrax from:", intergrax.__file__)
print("# files in selection:", len(SEL))
sys.stdout.flush()
t = time.time()
rc = pytest.main(list(SEL) + ["-q"])
print("# pytest exit code:", int(rc))
print("# elapsed_s: %.2f" % (time.time() - t))
sys.stdout.flush()
os._exit(int(rc))
