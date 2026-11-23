#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Sandbox isolation-strength gate (PDC-D6-R010, T-IGX-051-B Part H).

``check_sandbox_policy_wiring.py`` proves that a sandbox is wired for a product profile with
side-effect tools; this gate proves that a present but weak sandbox is refused. "Weak" is defined by
the strength attributes of the sandbox substrate's attested ``SandboxSecurityCapabilities``
(Q-SUP-07 draft default), with the table below:

==============================  =========================================
attribute                       required for a product exec-tool profile
==============================  =========================================
``isolation_tier``              ``container`` or ``cloud`` (not ``local``)
network egress                  deny or exact-allowlist enforcement proven
``supports_sandboxed_exec``     ``True`` (proven, not unattested)
``process_execution``           ``SANDBOXED`` when attested
==============================  =========================================

The gate composes a product profile with the exec tool enabled, checks that a weak capability set
is refused and a strong one is accepted, and exits 1 otherwise. In-process composition only; no
sandbox provider is contacted.
"""

from __future__ import annotations

import sys

from intergrax.applications._shared.sandbox_wiring import product_requires_sandbox, tool_profile_with_sandbox
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile, SandboxProfile
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities
from intergrax.runtime.sandbox.execution_environment import ProcessExecution

STRONG_ISOLATION_TIERS: frozenset[str] = frozenset({"container", "cloud"})


def exec_sandbox_required(env: ApplicationEnvironmentProfile) -> bool:
    """True for a product profile whose tool profile reaches side-effect or exec tools."""
    sandbox = env.sandbox
    if sandbox is not None and sandbox.enable_exec_tool:
        env = env.model_copy(update={"tool_profile": tool_profile_with_sandbox(env)})
    return product_requires_sandbox(env)


def isolation_strength_issues(
    env: ApplicationEnvironmentProfile,
    capabilities: SandboxSecurityCapabilities | None,
) -> tuple[str, ...]:
    """Reasons why the attested sandbox is too weak for ``env``; empty when it is acceptable."""
    if not exec_sandbox_required(env):
        return ()
    if env.sandbox is None:
        return ("sandbox_not_wired",)
    if capabilities is None:
        return ("sandbox_capabilities_unattested",)
    issues: list[str] = []
    if capabilities.isolation_tier not in STRONG_ISOLATION_TIERS:
        issues.append("isolation_tier_too_weak")
    if (
        capabilities.network_egress_deny_enforced is not True
        and capabilities.network_egress_allowlist_enforced is not True
    ):
        issues.append("network_egress_not_enforced")
    if capabilities.supports_sandboxed_exec is not True:
        issues.append("sandboxed_exec_not_proven")
    if capabilities.process_execution is not None and capabilities.process_execution is not ProcessExecution.SANDBOXED:
        issues.append("process_execution_not_sandboxed")
    return tuple(issues)


def weak_capabilities() -> SandboxSecurityCapabilities:
    """A present sandbox with a local tier and unproven egress: wired, but weak."""
    return SandboxSecurityCapabilities(
        isolation_tier="local",
        provider_id="local-process",
        network_egress_deny_enforced=None,
        supports_sandboxed_exec=True,
    )


def strong_capabilities() -> SandboxSecurityCapabilities:
    return SandboxSecurityCapabilities(
        isolation_tier="container",
        provider_id="container-fixture",
        network_egress_deny_enforced=True,
        supports_sandboxed_exec=True,
        process_execution=ProcessExecution.SANDBOXED,
    )


def product_exec_profile() -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.product_defaults(tool_ids=["sandbox.exec", "workspace.write_file"])
    return base.model_copy(update={"sandbox": SandboxProfile(enable_exec_tool=True)})


def main() -> int:
    env = product_exec_profile()
    if not exec_sandbox_required(env):
        print("a product exec-tool profile must require a sandbox", file=sys.stderr)
        return 1
    weak = isolation_strength_issues(env, weak_capabilities())
    if not weak:
        print("a present but weak sandbox must be refused", file=sys.stderr)
        return 1
    strong = isolation_strength_issues(env, strong_capabilities())
    if strong:
        print(f"a strong sandbox must be accepted: {', '.join(strong)}", file=sys.stderr)
        return 1
    if isolation_strength_issues(env, None) != ("sandbox_capabilities_unattested",):
        print("an unattested sandbox must be refused", file=sys.stderr)
        return 1
    print(f"OK: sandbox isolation strength (weak refused: {', '.join(weak)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
