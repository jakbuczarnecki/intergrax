import json, subprocess, sys
sys.path.insert(0, ".")
from scripts.governance.iiac_common import product_paths_changed_between as f, PRODUCT_PATH_PREFIXES
pin = "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"
P = "docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/"
a = set(f(pin, "fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f"))
b = set(f(pin, "19d552beeb6f0677a8bdc3b4ce65da15dd637c7b"))
h = set(f(pin, "ae5623efbdfd69ddf4ba67f82861ccad130c9b33"))
d = set(f("fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f", "19d552beeb6f0677a8bdc3b4ce65da15dd637c7b"))
mc = set(f("6a6279a49a9aa5a33936a1c6a577c7194b4bcebb", "b0e19bf61b346e6941b2e190b703c6922e708b34"))
s1 = json.load(open(P+"audit_scope.json", encoding="utf-8")); s2 = json.load(open(P+"addendum_1/audit_scope.json", encoding="utf-8"))
res = {
 "prefixes": list(PRODUCT_PATH_PREFIXES),
 "gate_paths_at_fa6ff1c96": len(a), "gate_paths_at_19d552bee": len(b), "gate_paths_at_head_ae5623efb": len(h),
 "union_fa6_and_19d": len(a|b), "head_equals_19d": h == b,
 "delta_fa6_to_19d": sorted(d), "delta_count": len(d), "delta_already_in_153": sorted(d & a), "delta_new": sorted(d - a),
 "scope1_equals_gate_at_fa6": sorted(s1["act_n11_paths"]) == sorted(a), "scope1_count": len(s1["act_n11_paths"]),
 "mcp_rc_paths_gate_count": len(mc), "mcp_rc_subset_of_153": mc <= a, "pdc_x_equals_153_minus_mcp_rc": set(s1["pdc_x_paths"]) == a - mc,
 "pdc_x_count": len(s1["pdc_x_paths"]), "scope2_delta_equals_git": sorted(s2["delta_act_n11_paths"]) == sorted(d),
 "scope2_delta_new_equals": sorted(s2["delta_paths_new_to_the_set"]) == sorted(d-a),
 "mcp_rc_byte_identical_b0e_vs_19d": subprocess.run(["git","diff","--quiet","b0e19bf61","19d552bee","--"]+sorted(mc)).returncode == 0,
 "mcp_rc_byte_identical_b0e_vs_head": subprocess.run(["git","diff","--quiet","b0e19bf61","ae5623efb","--"]+sorted(mc)).returncode == 0,
 "product_diff_19d_vs_head": len(f("19d552beeb6f0677a8bdc3b4ce65da15dd637c7b","ae5623efbdfd69ddf4ba67f82861ccad130c9b33")),
 "product_diff_fa6_vs_962": len(f("962c1d6706042675a7a2add6f4a89eb4d9deedb3","fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f")),
}
print(json.dumps(res, indent=1))
json.dump(res, open(".tmp/session/p7-00-cert/act_n11_union.json","w"), indent=1)
