"""Run the MCP-RC mandatory contract once; file list passed from Python, never via a text file."""
import os, subprocess, sys, time
sys.path.insert(0, os.getcwd())
from tests.qualification.mcp_server_x.rc_selection import MCP_SERVER_X_RC_SELECTION
files = list(MCP_SERVER_X_RC_SELECTION)
print(f"selection files: {len(files)}", flush=True)
t = time.time()
rc = subprocess.call([sys.executable, "-m", "pytest", *files, "-q"])
print(f"exit={rc} elapsed={time.time()-t:.1f}s", flush=True)
sys.exit(rc)
