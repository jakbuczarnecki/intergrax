"""IA-D1 adversarial probes A (no repo edit; run from the worktree root).

Usage: python .tmp/session/pdc-ia-d1/probes/probe_d1_a.py
Each probe prints PROBE <id> <observation>. Probes only call public service APIs on in-memory stacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "agents"), str(ROOT / "applications")]

from intergrax.agent_distribution.admin_models import RetireAgentRequest  # noqa: E402
from intergrax.agent_distribution.errors import *  # noqa: E402,F403
from intergrax.agent_distribution.installation import InstallationState  # noqa: E402
from intergrax.agent_distribution.trust import (  # noqa: E402
    AgentPackageTrustRevocationState,
)
from intergrax.agent_distribution.upgrade_admission import evaluate_agent_upgrade_admission  # noqa: E402
from intergrax.agent_distribution.identity import AgentPackageIdentity  # noqa: E402
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision  # noqa: E402
from intergrax.runtime.governance.control_plane_mutation_authorization import (  # noqa: E402
    ControlPlaneMutationAuthorizationBoundary,
)
from tests.qualification.pdc_x.d1 import _d1_harness as h  # noqa: E402

V1, V11 = "1.0.0", "1.1.0"


class _Deny:
    def evaluate(self, request):  # noqa: ANN001
        return PolicyDecision(action=PolicyAction.DENY, reason="probe_deny")


def out(pid: str, msg: str) -> None:
    print(f"PROBE {pid}: {msg}", flush=True)


def active(stack):
    a = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    return None if a is None else (a.installation_id, a.installation_state.value)


# ---- P1: same tenant, other application upgrades / installs into a slot it does not own
def p1() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(V1))
    cand = h.release(V11, contract_version="1.1.0")
    stack.register(cand)
    try:
        res = stack.service.upgrade_agent(
            application_id=h.APP_B,
            application_environment_id=h.ENV,
            request=cand.upgrade_request(expected_active=h.installation_id(V1)),
            principal=h.principal(),
        )
        out("P1-upgrade", f"app-b upgraded app-a slot outcome={res.outcome.value} active_now={active(stack)}")
    except Exception as exc:  # noqa: BLE001
        out("P1-upgrade", f"denied {type(exc).__name__}: {exc}")


# ---- P2: policy DENY boundary on upgrade/retire (tenant resolver passes, policy denies)
def p2() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(V1))
    cand = h.release(V11, contract_version="1.1.0")
    stack.register(cand)
    stack.service._mutation_authorization_boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_Deny())
    for name, call in (
        (
            "upgrade",
            lambda: stack.service.upgrade_agent(
                application_id=h.APP,
                application_environment_id=h.ENV,
                request=cand.upgrade_request(expected_active=h.installation_id(V1)),
                principal=h.principal(),
            ),
        ),
        (
            "retire",
            lambda: stack.service.retire_agent(
                application_id=h.APP,
                application_environment_id=h.ENV,
                request=RetireAgentRequest(
                    mutation_id="m-r", installation_slot_id=h.SLOT, expected_active_installation_id=h.installation_id(V1)
                ),
                principal=h.principal(),
            ),
        ),
    ):
        try:
            call()
            out(f"P2-{name}", f"NOT DENIED active={active(stack)}")
        except Exception as exc:  # noqa: BLE001
            out(f"P2-{name}", f"denied {type(exc).__name__}; active={active(stack)}; installs={sorted(stack.state.installations)}")


# ---- P3: upgrade admission SemVer policy (pre-release, 0.x)
def p3() -> None:
    def ident(v: str, d: str = "a") -> AgentPackageIdentity:
        return AgentPackageIdentity(distribution_package_id="p", package_version=v, package_digest="sha256:" + d * 64)

    for prior, cand in (
        ("1.2.3", "1.2.4-rc.1"),
        ("1.2.3-rc.1", "1.2.3"),
        ("1.2.3", "1.2.3+build9"),
        ("0.1.0", "0.9.0"),
        ("1.0.0", "1.1.0"),
    ):
        rep = evaluate_agent_upgrade_admission(
            prior_identity=ident(prior, "a"),
            candidate_identity=ident(cand, "b"),
            prior_contracts={},
            candidate_contracts={},
            bindings=[],
        )
        out("P3", f"{prior} -> {cand}: admitted={rep.admitted} reason={rep.primary_reason}")


# ---- P4/P5: exception paths in upgrade_agent
def p4() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(V1))
    cand = h.release(V11, contract_version="1.1.0")
    stack.register(cand)

    def boom(*a, **k):  # noqa: ANN002, ANN003
        raise RuntimeError("probe: contract store unavailable")

    stack.service._resolve_package_contracts = boom
    try:
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=cand.upgrade_request(expected_active=h.installation_id(V1)),
            principal=h.principal(),
        )
    except Exception as exc:  # noqa: BLE001
        rec = stack.installation_store.get_installation(h.installation_id(V11))
        out(
            "P4",
            f"exc={type(exc).__name__}; candidate_state={rec.installation_state.value if rec else None}; active={active(stack)}",
        )
    # P5: promotion failure -> FAILED_CANDIDATE persisted, evidence discarded
    stack2 = h.build_stack()
    h.install_bind_enable(stack2, h.release(V1))
    stack2.register(cand)

    real = stack2.installation_service.promote_verified_to_active

    def bad_promote(*a, **k):  # noqa: ANN002, ANN003
        raise RuntimeError("probe: store conflict")

    stack2.installation_service.promote_verified_to_active = bad_promote
    try:
        stack2.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=cand.upgrade_request(expected_active=h.installation_id(V1)),
            principal=h.principal(),
        )
    except Exception as exc:  # noqa: BLE001
        rec = stack2.installation_store.get_installation(h.installation_id(V11))
        out(
            "P5",
            f"exc={type(exc).__name__}; candidate_state={rec.installation_state.value}; "
            f"evidence_returned_to_caller=NONE(exception carries no audit_event_types: {hasattr(exc, 'audit_event_types')})",
        )
    stack2.installation_service.promote_verified_to_active = real


# ---- P6: BLOCKED_NO_SAFE_TARGET: is new execution admission still open on the revoked serving revision?
def p6() -> None:
    from tests.qualification.pdc_x.d1 import test_d1_adversarial_lifecycle as t

    stack = t._stack_with_validated_revision()
    t._serving_two_revisions(stack)
    stack.revocation.state = AgentPackageTrustRevocationState(
        revoked_package_digests=frozenset({h.DIGESTS[V1], h.DIGESTS[V11]})
    )
    resp = stack.service.respond_to_emergency_revocation(application_id=h.APP, application_environment_id=h.ENV)
    out("P6-response", f"action={resp.action.value} reason={resp.response_reason_code.value}")
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    out("P6-serving", f"traffic_serving_revision_id={serving.traffic_serving_revision_id}")
    try:
        r = t._admit(stack, "rev-2")
        out("P6-admission", f"NEW EXECUTION ADMISSION STILL OPEN on revoked revision rev-2 -> {type(r).__name__}")
    except Exception as exc:  # noqa: BLE001
        out("P6-admission", f"closed: {type(exc).__name__}")


# ---- P7: replay returns before authorization
def p7() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(V1))
    cand = h.release(V11, contract_version="1.1.0")
    stack.register(cand)
    req = cand.upgrade_request(expected_active=h.installation_id(V1))
    stack.service.upgrade_agent(
        application_id=h.APP, application_environment_id=h.ENV, request=req, principal=h.principal()
    )
    stack.service._mutation_authorization_boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_Deny())
    try:
        res = stack.service.upgrade_agent(
            application_id=h.APP, application_environment_id=h.ENV, request=req, principal=h.principal()
        )
        out("P7-upgrade-replay", f"policy-DENY principal still got outcome={res.outcome.value} (view of slot state)")
    except Exception as exc:  # noqa: BLE001
        out("P7-upgrade-replay", f"denied {type(exc).__name__}")
    rr = RetireAgentRequest(
        mutation_id="m-r", installation_slot_id=h.SLOT, expected_active_installation_id=h.installation_id(V11)
    )
    stack.service._mutation_authorization_boundary = h.allow_mutation_boundary()
    stack.service.retire_agent(
        application_id=h.APP, application_environment_id=h.ENV, request=rr, principal=h.principal()
    )
    stack.service._mutation_authorization_boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_Deny())
    try:
        res = stack.service.retire_agent(
            application_id=h.APP, application_environment_id=h.ENV, request=rr, principal=h.principal()
        )
        out("P7-retire-replay", f"policy-DENY principal still got already_done={res.already_done}")
    except Exception as exc:  # noqa: BLE001
        out("P7-retire-replay", f"denied {type(exc).__name__}")


# ---- P8: forged trust record on direct install (no signature ever verified at install_agent)
def p8() -> None:
    from intergrax.agent_distribution.trust import AgentQualificationEvidenceKind, AgentTrustEvidenceRef

    stack = h.build_stack()
    rel = h.release(V1)
    stack.register(rel)
    req = rel.install_request()
    forged = req.trust_record.model_copy(
        update={
            "trust_evidence_refs": (
                AgentTrustEvidenceRef(evidence_id="forged:not-a-real-signature", kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION),
            )
        }
    )
    res = stack.service.install_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=req.model_copy(update={"trust_record": forged}),
        principal=h.principal(),
    )
    out("P8", f"install with caller-asserted SIGNATURE_VERIFICATION ref and no verifier call -> state={res.installation.installation_state}")


if __name__ == "__main__":
    for fn in (p1, p2, p3, p4, p6, p7, p8):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            out(fn.__name__, f"PROBE ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
