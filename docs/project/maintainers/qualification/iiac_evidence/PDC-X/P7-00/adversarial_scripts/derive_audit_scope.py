"""Derive the P7-00 audit scope from git only (auditor-owned)."""
import json, subprocess, sys

PIN = "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"
AUD = subprocess.check_output(["git", "rev-parse", "fa6ff1c96"], text=True).strip()
MCPA, MCPB = "6a6279a49", "b0e19bf61"

def git(*a):
    return subprocess.check_output(["git", *a], text=True).strip().splitlines()

all_paths = git("diff", "--name-only", PIN, AUD, "--", "intergrax", "agents", "applications")
mcp = git("diff", "--name-only", MCPA, MCPB, "--", "intergrax", "applications", "agents")
mcp_set = set(mcp)
in_all = [p for p in mcp if p in set(all_paths)]
pdc = sorted(set(all_paths) - mcp_set)
status = dict(l.split("\t", 1) for l in git("diff", "--name-status", PIN, AUD, "--", "intergrax", "agents", "applications"))
byname = {}
for l in git("diff", "--name-status", PIN, AUD, "--", "intergrax", "agents", "applications"):
    s, p = l.split("\t", 1)
    byname[p] = s
drift = subprocess.run(["git", "diff", "--quiet", MCPB, AUD, "--", *mcp]).returncode
numstat = git("diff", "--numstat", PIN, AUD, "--", *pdc)
ins = sum(int(l.split("\t")[0]) for l in numstat); dele = sum(int(l.split("\t")[1]) for l in numstat)
commits = git("log", "--no-merges", "--format=%h %s", f"{PIN}..{AUD}", "--", *pdc)
scope = {
    "stage_id": "PDC-X/P7-00",
    "session_designation": "PDC-P7-00-ACT-N11-BASELINE-G6-AUDIT-2026-11-13",
    "certification_profile": "iiac-v1",
    "policy_version": "IIAC-2026-11-05.1",
    "campaign": "PDC-P7-00 ACT-N11 activation audit: integrated baseline (mcp-rc) vs activation pin",
    "ticket_anchors": ["governance/tickets/T-IGX-038-F.md"],
    "activation_pin_audit_execution_sha": PIN,
    "audited_implementation_sha": AUD,
    "audited_product_tree": "962c1d6706042675a7a2add6f4a89eb4d9deedb3 (fa6ff1c96 adds only an evidence log)",
    "integrated_baseline_sha": "87f4d4f417050fec40c32e4051bf76fb1124d58b (origin/development, merge of PR #520)",
    "auditor_branch": "p7-00-baseline-audit",
    "act_n11_paths_total": len(all_paths),
    "act_n11_paths": sorted(all_paths),
    "mcp_rc_paths_count": len(mcp),
    "mcp_rc_paths_all_in_act_n11": len(in_all) == len(mcp),
    "mcp_rc_paths": sorted(mcp),
    "mcp_rc_byte_identical_to_b0e19bf61": drift == 0,
    "mcp_rc_reaudit_reference": "docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md (PASS_SCOPED at b0e19bf61)",
    "pdc_x_paths_count": len(pdc),
    "pdc_x_paths": pdc,
    "pdc_x_path_status": {p: byname[p] for p in pdc},
    "pdc_x_diff_lines": {"insertions": ins, "deletions": dele},
    "pdc_x_commits_touching_paths": commits,
    "split_rule": "mcp_rc = git diff --name-only 6a6279a49 b0e19bf61 -- intergrax applications agents; pdc_x = act_n11 minus mcp_rc",
    "out_of_scope": [
        "eight per-domain PDC-P7 audits IA-D1..IA-D8",
        "CI contexts / GitHub Actions",
        "P8 (MCP-SERVER-X) edits and FRZ counter 225",
        "G-DOC-01..12, documentation-only changes",
    ],
}
json.dump(scope, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
print(len(all_paths), len(mcp), len(pdc), drift, ins, dele)
