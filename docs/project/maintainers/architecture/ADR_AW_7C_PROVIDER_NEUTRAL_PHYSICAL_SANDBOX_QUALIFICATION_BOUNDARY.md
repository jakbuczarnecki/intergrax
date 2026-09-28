# ADR-AW-7C-PROVIDER-NEUTRAL-PHYSICAL-SANDBOX-QUALIFICATION-BOUNDARY

| Field | Value |
|-------|-------|
| **Status** | **Accepted** — AW-7C-P0-3B architecture/qualification correction |
| **Date** | 2026-09-28 |
| **Baseline ancestor** | `a59744517b92847f55def1db22826d17d89ee155` |
| **Related** | AW-7C · FRZ-SEC-05 · FRZ-SEC-06 · FRZ-PRD-02 · PROD-Q |

---

## 1. Decision

AW-7C **platform capability qualification** requires a **physical** enforcement proof on a **controlled, provider-neutral reference substrate**. It does **not** semantically require E2B, Modal, Daytona, or any other external hosted sandbox SaaS.

**Capability qualification** and **provider qualification** are distinct lifecycle gates.

---

## 2. Capability qualification (AW-7C)

- Proves Intergrax sandbox/network egress **contract and enforcement semantics**.
- May use a **qualification-only reference substrate** (real Linux network namespace, kernel firewall, isolated process execution).
- Must use **real OS/substrate enforcement** — not metadata-only or mock proof.
- Traverses the existing hosted execution channel: `HostedSandboxSession` → `SandboxHostBackend` → qualification reference backend → kernel policy.

---

## 3. Provider qualification (PROD-Q)

- Proves a **concrete production provider** (E2B, Modal, Daytona, future) correctly implements the already-qualified platform contract.
- **Mandatory before production activation** of that provider.
- Belongs to **PROD-Q / FRZ-PRD-02**, not AW-7C closure.

E2B adapter remains in-repo; **E2B physical provider qualification = DEFERRED TO PROD-Q** and is **not** an AW-7C prerequisite.

Modal and Daytona remain possible production providers; their provider-specific qualification is **not** required to close AW-7C capability qualification.

---

## 4. Reference substrate boundaries (hard)

The reference substrate **must not** become:

- a production provider;
- a second Integration Catalog entry;
- a provider-selection authority;
- a Runtime production fallback;
- an automatically selectable sandbox backend.

Production behavior remains **fail closed** when no independently qualified production provider satisfies required security capabilities.

---

## 5. Security attestation

Reference attestation is derived from **independently verified kernel state** (effective nftables/iptables in the sandbox network namespace), correlated with runtime probes. **Requested policy echoed as “enforced” is rejected** as qualification evidence.

This decision does **not** weaken: `SandboxSecurityRequirements`, `SandboxSecurityCapabilities`, `NetworkEgressAllowlist`, exact-host scope, fail-closed substrate selection, or Governance / Execution ownership boundaries.

---

## 6. Threat-model boundary

**Established by reference qualification:** process isolation boundary; kernel egress enforcement; allowed destination reachable; denied destination alive but unreachable under policy; HTTP redirect cannot widen scope; policy evidence matches applied substrate state.

**Not claimed:** external SaaS provider correctness; public DNS rebinding/CDN/IP churn for every provider; cloud isolation guarantees; hostile kernel escape; provider IAM/account security — these remain **provider/infrastructure qualification** (PROD-Q).

---

## 7. Tenant isolation

Reference qualification uses synthetic test attribution only — **N/A — WITH EVIDENCE** for tenant certification. This does **not** certify production tenant semantics (no FRZ-TEN PASS).

---

## 8. Consequences

- AW-7C-P0-3B delivers provider-neutral physical proof under `tests/integration/runtime/sandbox/reference_substrate/`.
- Canonical qualification docs and roadmap record E2B physical qualification as **DEFERRED / NOT ESTABLISHED** for AW-7C while preserving historical E2B harness evidence.
- **Zero production Python** changes required for this boundary correction.
