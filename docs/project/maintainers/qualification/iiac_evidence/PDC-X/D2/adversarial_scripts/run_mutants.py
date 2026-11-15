"""IA-D2 runtime mutants: monkeypatch one D2 guard per process, run the claiming D2 tests, expect FAIL.

No product file is edited. Each mutant runs in a fresh interpreter; run sequentially (one pytest at a time).
Usage: python run_mutants.py [mutant_id ...]
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
PY = Path(sys.executable)
D2 = "tests/qualification/pdc_x/d2"

MUTANTS: dict[str, tuple[str, list[str], str]] = {
    "M1-hard-trim-drops-identity": (
        "R011",
        [f"{D2}/test_d2_context_fidelity.py", "-k", "hard_trim"],
        """
        import intergrax.runtime.nexus.context.degradation_ladder as m
        from intergrax.llm.messages import ChatMessage
        def bad_replace(message, **kw):
            return ChatMessage(role=message.role, content=kw.get("content", message.content))
        m.replace = bad_replace
        """,
    ),
    "M2-no-atomic-group-drop": (
        "R011",
        [f"{D2}/test_d2_context_fidelity.py", "-k", "orphan or atomically"],
        """
        import intergrax.runtime.nexus.context.degradation_ladder as m
        import intergrax.runtime.nexus.context.context_compiler as c
        m.complete_tool_call_group_spans = lambda msgs: ()
        c.complete_tool_call_group_spans = lambda msgs: ()
        """,
    ),
    "M3-egress-filter-noop": (
        "R014",
        [f"{D2}/test_d2_context_fidelity.py", "-k", "sensitive"],
        """
        import intergrax.runtime.nexus.context.context_engine as m
        m.filter_fragments_by_provider_egress = lambda fragments, **k: (fragments, [])
        """,
    ),
    "M4-unlinked-tool-output-allowed": (
        "R009",
        [f"{D2}/test_d2_context_fidelity.py", "-k", "tool_output"],
        """
        import inspect, sys
        import intergrax.runtime.nexus.context.context_engine
        import intergrax.context.provider_lifecycle as m
        orig = m.canonicalize_fragment
        src = inspect.getsource(m)
        marker = "if fragment.source is ContextFragmentSource.TOOL_OUTPUT:"
        assert marker in src
        ns = dict(m.__dict__)
        exec(compile(src.replace(marker, "if False:"), m.__file__, "exec"), ns)
        mutant = ns["canonicalize_fragment"]
        for mod in list(sys.modules.values()):
            if getattr(mod, "canonicalize_fragment", None) is orig:
                mod.canonicalize_fragment = mutant
        """,
    ),
    "M5-retention-never-due": (
        "R015",
        [f"{D2}/test_d2_memory_retention.py"],
        """
        import intergrax.memory.retention_sweep as m
        m._expiry_instant = lambda *a, **k: None
        """,
    ),
    "M6-freshness-always-fresh": (
        "R005",
        [f"{D2}/test_d2_rag_freshness.py", f"{D2}/test_d2_rag_freshness_wiring.py"],
        """
        import intergrax.rag.governance.source_freshness as m
        def fresh(chunk, *, current):
            return m.SourceFreshnessVerdict(source_id=current.source_id, status=m.SourceFreshnessStatus.FRESH, reason="mutant")
        m.evaluate_source_freshness = fresh
        """,
    ),
    "M7-mandatory-base-never-protected": (
        "R007",
        [f"{D2}/test_d2_context_fidelity.py", "-k", "mandatory"],
        """
        import intergrax.runtime.nexus.context.context_compiler as c
        import intergrax.runtime.nexus.context.degradation_ladder as d
        import intergrax.context.budget.mandatory_base_messages as b
        c.mandatory_base_message_indices = lambda *a, **k: frozenset()
        c.mandatory_protected_message_indices = lambda *a, **k: frozenset()
        c.estimate_mandatory_protected_message_tokens = lambda *a, **k: 0
        """,
    ),
    "M8-sweep-skips-vector-tombstone": (
        "R015",
        [f"{D2}/test_d2_memory_retention.py", "-k", "tombstoned or flaky or failed_vector"],
        """
        import intergrax.memory.default_memory_control_plane as m
        orig = m.DefaultMemoryControlPlane._forget_user
        async def noop(self, identity, scope, request):
            res = await orig(self, identity, scope, request)
            import dataclasses
            return dataclasses.replace(res, lifecycle=None)
        m.DefaultMemoryControlPlane._forget_user = noop
        """,
    ),
    "M9-tenant-bound-guard-noop": (
        "R002",
        ["tests/qualification/pdc_x/xd/test_xd_02_agent_context.py"],
        """
        import intergrax.memory.default_memory_control_plane as m
        m.DefaultMemoryControlPlane._assert_identity_tenant_is_bound_tenant = lambda self, identity: None
        """,
    ),
}


def run(mutant_id: str) -> tuple[str, int, str]:
    row, args, patch = MUTANTS[mutant_id]
    code = textwrap.dedent(patch) + "\nimport sys, pytest\nsys.exit(pytest.main(" + repr(["-q", "-x", "--no-header", "-p", "no:randomly", *args]) + "))\n"
    out_dir = ROOT / ".tmp" / "session" / "ia-d2" / "mutants"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(PY), "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    log = out_dir / f"{mutant_id}.log"
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1:] or [""]
    return row, proc.returncode, tail[0]


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(MUTANTS)
    for mid in wanted:
        row, rc, tail = run(mid)
        verdict = "KILLED" if rc != 0 else "SURVIVED"
        print(f"{mid} [{row}] rc={rc} {verdict} :: {tail}", flush=True)
