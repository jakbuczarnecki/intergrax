"""IA-D6 probes, part 2 (auditor-owned, read-only)."""
from __future__ import annotations

import asyncio
import json
import traceback

R: dict[str, object] = {}


def probe(name):
    def deco(fn):
        try:
            R[name] = fn()
        except Exception as e:  # noqa: BLE001
            R[name] = {"probe_error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-500:]}
        return fn

    return deco


@probe("TS1_tenant_scope_unscoped_resource_is_accessible")
def _():
    from intergrax.runtime.security.tenant_scope import tenant_scope_is_valid as v

    return {
        "req_a_res_none": v("a", None, allow_unscoped=False),
        "req_a_res_blank": v("a", "  ", allow_unscoped=False),
        "req_none_res_a": v(None, "a", allow_unscoped=False),
        "req_a_res_A": v("a", "A", allow_unscoped=False),
        "none_none_strict": v(None, None, allow_unscoped=False),
    }


@probe("TS2_middleware_intake_resource_tenant_none")
def _():
    from intergrax.applications._shared.application_security_wiring import TenantSecurityMiddleware
    from intergrax.contracts.execution_phase import ExecutionPhase
    from intergrax.runtime.hooks.hook_point import HookPoint
    from tests.support.middleware_hook_test_context import tenant_intake_hook_context_for_test

    m = TenantSecurityMiddleware()
    out = {}
    for label, res in (("resource_none", None), ("resource_other", "tenant-b"), ("resource_same", "tenant-a")):
        ctx = tenant_intake_hook_context_for_test(
            tenant_id="tenant-a", resource_tenant_id=res, user_id="u", phase=ExecutionPhase.INTAKE
        )
        out[label] = str(asyncio.run(m.before(HookPoint.BEFORE_TASK_INTAKE, ctx)).action)
    return out


@probe("G1_raw_secret_gate_scope_and_extra_evasions")
def _():
    from intergrax.core.security.raw_secret_gate import scan_source_for_raw_secrets as s

    cases = {
        "env_or_literal": "import os\nk = os.environ.get('K') or 'harness-dev-signing-key'\n",
        "walrus_default": "import os\nif (k := os.environ.get('K')) is None:\n    k = 'harness-dev-signing-key'\n",
        "dict_get_default": "d = {}\nsigning_secret = d.get('k', 'harness-dev-signing-key')\n",
        "kwarg_literal": "def f(api_key='abcdEFGH12345678abcd'): pass\n",
        "named_literal": "signing_key = 'harness-dev-signing-key'\n",
        "concat": "signing_key = 'harness-dev-' + 'signing-key'\n",
        "aws_token_in_dict": "cfg = {'x': 'AKIAIOSFODNN7EXAMPLE'}\n",
        "aws_token_assign": "x = 'AKIAIOSFODNN7EXAMPLE'\n",
        "bearer_in_fstring": "x = f'Bearer eyJhbGciOiJIUzI1NiJ9.aaaaaaaaaa.bbbbbbbbbb'\n",
    }
    return {k: [f.rule if hasattr(f, 'rule') else str(f) for f in s(v, path='probe.py')] for k, v in cases.items()}


@probe("SIG1_signing_has_no_production_consumer")
def _():
    import pathlib
    import re

    hits = []
    for root in ("intergrax", "agents", "applications"):
        for p in pathlib.Path(root).rglob("*.py"):
            t = p.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"\bsign_critical_action\(|\bverify_critical_action_signature\(", t):
                hits.append(p.as_posix())
    return sorted(hits)


print(json.dumps(R, indent=1, default=str))
