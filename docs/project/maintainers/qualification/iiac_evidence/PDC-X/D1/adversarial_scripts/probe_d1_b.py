"""IA-D1 adversarial probes B (rollback to revoked, skill lifecycle, discovery gate bypass, registry bridge).

Usage: python .tmp/session/pdc-ia-d1/probes/probe_d1_b.py (from the worktree root).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "agents"), str(ROOT / "applications")]

from intergrax.agent_distribution.admin_models import RollbackRuntimeRevisionRequest, RetireAgentRequest  # noqa: E402
from intergrax.agent_distribution.trust import AgentPackageTrustRevocationState  # noqa: E402
from tests.qualification.pdc_x.d1 import _d1_harness as h  # noqa: E402

V1, V11 = "1.0.0", "1.1.0"


def out(pid: str, msg: str) -> None:
    print(f"PROBE {pid}: {msg}", flush=True)


# ---- P9: operator rollback_revision to a revision whose package digest is revoked after the fact
def p9() -> None:
    from tests.qualification.pdc_x.d1 import test_d1_adversarial_lifecycle as t

    stack = t._stack_with_validated_revision()
    t._serving_two_revisions(stack)  # rev-1 (v1.0.0) prior, rev-2 (v1.1.0) serving
    stack.revocation.state = AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[V1]}))
    try:
        res = stack.service.rollback_revision(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=RollbackRuntimeRevisionRequest(
                mutation_id="m-rb",
                expected_current_traffic_revision_id="rev-2",
                expected_serving_pointer_revision=2,
            ),
            principal=h.principal(),
        )
        out("P9", f"operator rollback restored REVOKED package revision: traffic={res.traffic_serving_revision_id}")
    except Exception as exc:  # noqa: BLE001
        out("P9", f"refused {type(exc).__name__}: {str(exc)[:120]}")


# ---- P10: skill lifecycle
def p10() -> None:
    from tests.qualification.pdc_x.d1 import test_d1_skill_lifecycle as s

    names = [n for n in dir(s) if not n.startswith("__")]
    out("P10-names", ",".join(n for n in names if n.startswith("_") and not n.startswith("__"))[:300])


# ---- P11: discovery owner gate bypass (static): which syntaxes does the gate's scanner miss?
def p11() -> None:
    from tests.qualification.pdc_x.d1 import test_d1_discovery_owner as g

    samples = {
        "direct import": "from importlib.metadata import entry_points\nentry_points(group='x')",
        "metadata.entry_points": "import importlib.metadata\nimportlib.metadata.entry_points(group='x')",
        "alias md": "import importlib.metadata as md\nmd.entry_points(group='x')",
        "from importlib import metadata as md": "from importlib import metadata as md\nmd.entry_points(group='x')",
        "importlib_metadata backport": "import importlib_metadata\nimportlib_metadata.entry_points(group='x')",
        "distributions scan": "import importlib.metadata as md\n[d.entry_points for d in md.distributions()]",
        "pkg_resources": "import pkg_resources\nlist(pkg_resources.iter_entry_points('x'))",
        "getattr indirection": "import importlib.metadata as m\ngetattr(m, 'entry_' + 'points')(group='x')",
    }
    for label, src in samples.items():
        out("P11", f"{label!r}: gate detects={g._scans_entry_points(ast.parse(src))}")


if __name__ == "__main__":
    for fn in (p9, p11):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            out(fn.__name__, f"PROBE ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
