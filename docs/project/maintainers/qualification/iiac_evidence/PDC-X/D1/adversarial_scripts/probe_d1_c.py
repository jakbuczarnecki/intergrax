"""IA-D1 adversarial probes C (skill lifecycle, activation after revocation, emergency entry point authority)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "agents"), str(ROOT / "applications")]

from intergrax.agent_distribution.emergency_revocation_response import AgentEmergencyRevocationRequest  # noqa: E402
from intergrax.agent_distribution.trust import AgentPackageTrustRevocationState  # noqa: E402
from tests.qualification.pdc_x.d1 import _d1_harness as h  # noqa: E402
from tests.qualification.pdc_x.d1 import test_d1_skill_lifecycle as sk  # noqa: E402

V1, V11 = "1.0.0", "1.1.0"


def out(pid: str, msg: str) -> None:
    print(f"PROBE {pid}: {msg}", flush=True)


class _Abort(BaseException):
    pass


def p12() -> None:
    # F-11: BaseException during upgrade materialization
    life = sk._installed()
    mat = sk._Materializer(life.registry)

    def boom(res):  # noqa: ANN001
        raise _Abort()

    mat.materialize = boom  # type: ignore[method-assign]
    try:
        life.upgrade_skill(operation_id="op-u", host_profile_id=sk._HOST, resolved=sk._resolution("1.1.0", "b"), materializer=mat)
    except _Abort:
        out("P12-F11", f"BaseException during upgrade -> skill still installed={life.is_bound(sk._SKILL)}")
    # control: Exception
    life2 = sk._installed()
    mat2 = sk._Materializer(life2.registry, fail=True)
    res = life2.upgrade_skill(operation_id="op-u", host_profile_id=sk._HOST, resolved=sk._resolution("1.1.0", "b"), materializer=mat2)
    out("P12-ctl", f"Exception control -> {res.outcome.value}; installed={life2.is_bound(sk._SKILL)}")
    # F-12: pre-release candidate replaces release
    life3 = sk._installed()
    res = sk._upgrade(life3, sk._resolution("1.0.1-rc.1", "c"))
    out("P12-F12", f"1.0.0 -> 1.0.1-rc.1 outcome={res.outcome.value} detail={res.reason_detail}")
    life4 = sk._installed()
    # skill upgrade does not verify anything about candidate beyond digest presence / identity
    res = sk._upgrade(life4, sk._resolution("1.0.1", "d"))
    out("P12-ok", f"1.0.0 -> 1.0.1 outcome={res.outcome.value}")


def p13() -> None:
    # activation of a validated revision whose package was revoked after build
    from tests.qualification.pdc_x.d1 import test_d1_adversarial_lifecycle as t

    stack = t._stack_with_validated_revision()
    stack.revocation.state = AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[V1]}))
    try:
        res = h.activate_revision(stack, "rev-1")
        out("P13", f"activation of revision pinning REVOKED package succeeded: traffic={res.traffic_serving_revision_id}")
    except Exception as exc:  # noqa: BLE001
        out("P13", f"refused {type(exc).__name__}: {str(exc)[:100]}")


def p14() -> None:
    # emergency revocation: no principal, caller-supplied revocation snapshot forces a rollback of a healthy revision
    from tests.qualification.pdc_x.d1 import test_d1_adversarial_lifecycle as t

    stack = t._stack_with_validated_revision()
    t._serving_two_revisions(stack)  # rev-2 serving (v1.1.0), rev-1 prior
    forged = AgentEmergencyRevocationRequest(
        application_id=h.APP,
        application_environment_id=h.ENV,
        evaluated_at=datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC),
        revocation_state=AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[V11]})),
    )
    # the REAL source says nothing is revoked; the caller (no principal, no tenant scope, no policy check) overrides it
    assert not stack.revocation.state.revoked_package_digests
    resp = stack.service.respond_to_emergency_revocation(application_id=h.APP, application_environment_id=h.ENV, request=forged)
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    out("P14", f"anonymous caller with forged snapshot -> action={resp.action.value}; traffic now {serving.traffic_serving_revision_id}")


if __name__ == "__main__":
    for fn in (p12, p13, p14):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            out(fn.__name__, f"PROBE ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
