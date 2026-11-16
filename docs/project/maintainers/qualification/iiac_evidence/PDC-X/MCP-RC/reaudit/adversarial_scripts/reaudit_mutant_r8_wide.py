"""R8 re-run against every catalog-related suite (does any existing test pin invoke_catalog_tool sanitisation?)."""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
REPO = Path(os.environ.get("AUDIT_ROOT", Path.cwd())).resolve()  # the checkout under audit
sys.path.insert(0, str(REPO))
import scripts.qualification.mcp_rc_mutation as h
tests = tuple(sorted(p.replace("\\", "/") for p in glob.glob(str(REPO / "tests/unit/applications/test_mcp_catalog*.py")))) 
tests = tuple(Path(t).relative_to(REPO).as_posix() for t in tests) + (
    "tests/qualification/mcp_server_x/test_mcp_server_x_p7_l3_invoke_security.py",
    "tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_security_trace.py",
    "tests/unit/applications/test_mcp_rc_audit_corrections.py",
)
h.MUTANTS = (h.Mutant("R8w-invoke-catalog-tool-unsanitized-wide", "invoke_catalog_tool never echoes raw exception text (all catalog suites)",
    f"{h._SHARED}/mcp_catalog_tools.py",
    'async with optional_mcp_call_admission(call_admission), sanitized_tool_errors("invoke_catalog_tool"):',
    "async with optional_mcp_call_admission(call_admission):", tests),)
if __name__ == "__main__":
    raise SystemExit(h.main(sys.argv[1:]))
