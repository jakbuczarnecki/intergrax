# Intergrax Platform Proof Authoring Guide

**Status:** Canonical  
**Audience:** Independent proof-author sessions

Follow this guide to add or advance a platform proof without reconstructing methodology from scratch.

---

## Prerequisites

- Branch: `development` (resolve current shared HEAD before material work)
- Read: [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md)
- Check: [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md) for existing coverage

**Never:** classify a product (`applications/`) as a platform domain or migrate product proofs into `platform_proofs/`.

---

## Workflow

| Step | Action |
|------|--------|
| **1** | Resolve current `development` HEAD (`git fetch` + `git pull --ff-only`) |
| **2** | Identify canonical **DOMAIN** or **FEATURE** owner from [runtime architecture hub](../docs/project/architecture/intergrax_runtime_architecture.md) |
| **3** | Read owning architecture + plan only — no repo-wide audit |
| **4** | Define one **falsifiable claim** (see protocol § E) |
| **5** | Identify **real boundary** — do not mock the mechanism under proof |
| **6** | Design **positive and negative** scenarios (protocol § I) |
| **7** | Check existing `scripts/proof/` manifest and runner — reuse, do not duplicate |
| **8** | Implement the **smallest** proof that exercises the claim |
| **9** | Add package `proof.json` (`intergrax.platform_proof_descriptor.v1`) — descriptor-backed Platform Proofs are discovered automatically (no central manifest registration) |
| **10** | Run targeted deterministic validation (unit/integration gates) |
| **11** | Run the actual proof via canonical runner |
| **12** | Record evidence (`SuiteReceipt`; SHA, profile, result, limitations) |
| **12b** | For `evidence_required=true`, write `evidence.json` to runner-provided `INTERGRAX_PROOF_ARTIFACT_DIR` when executed via suite |
| **13** | Update [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md) coverage |
| **14** | Update [PROOFS.md](../docs/project/proofs/PROOFS.md) **only** if accepted public evidence/claim changes |

---

## Proof author checklist

- [ ] Claim is single, bounded, falsifiable
- [ ] Architecture owner identified (domain or feature)
- [ ] Real boundary named; mechanism under proof not mocked
- [ ] At least one negative / counterexample path designed
- [ ] PASS invariants are machine-checkable where possible
- [ ] FAIL conditions explicit
- [ ] Limitations and excluded claims documented
- [ ] `proof_id` unique across suite (descriptor-backed proofs self-register via `proof.json`)
- [ ] Package includes static `proof.json` descriptor (see protocol § D2)
- [ ] Execution uses `scripts/proof/` runner — no duplicate infrastructure
- [ ] Evidence uses `SuiteReceipt` — not merged with domain `ProofReceipt`
- [ ] For descriptor-backed proofs with `evidence_required=true`, PASS requires validated `evidence.json` (exit code alone is insufficient)
- [ ] Map coverage updated (`NO_PROOF` → `DESIGNED` → `EXECUTABLE` → `QUALIFIED`)
- [ ] Product proofs not cited as platform domain evidence
- [ ] Public dashboard updated only when public claim boundary changes

---

## Coverage vocabulary (map only)

| Label | Meaning |
|-------|---------|
| **NO_PROOF** | No canonical platform proof designed |
| **DESIGNED** | Claim/scenario defined; executable proof absent |
| **EXECUTABLE** | Registered/runnable proof exists in manifest |
| **QUALIFIED** | Successfully executed with accepted evidence under named bounded environment |

Coverage ≠ `ProofStatus` ≠ public claim status ≠ production maturity.

---

## Anti-patterns

| Anti-pattern | Why it fails |
|--------------|--------------|
| Fake replacing mechanism under proof | Cannot claim that boundary proved |
| Product proof masquerading as platform proof | Violates ownership rule |
| Proof buried only in `applications/` | Product-owned; not platform proof |
| Duplicate proof runner or manifest | Fragments execution truth |
| Undocumented environment assumptions | BLOCKED or false PASS |
| Claim broader than evidence | Public governance violation |
| No negative scenario | Not falsification — demo only |
| PASS based only on prose | Not machine-checkable |
| PASS based only on child exit code when `evidence_required=true` | Suite verifies typed `evidence.json` via `PlatformProofEvidence` |
| Chain-of-thought collection as evidence | Not platform invariant |
| Reimplementing platform components inside proof | Proves clone, not platform |
| Copying production product business logic | Product proof, not platform proof |

---

## Reference: first designed proof

**`TOOLS-ITERATIVE-SQL-INVESTIGATION`** — see [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md) and [tools/README.md](tools/README.md).

Planned scenarios (design only until implemented):

- **A.** multi-hop anomaly investigation  
- **B.** correlation ≠ causation  
- **C.** missing evidence → bounded limitation  

Real boundaries: real PostgreSQL, real model, real tool runtime, ENG-5 investigation policy, ENG-6 InvestigationProof.

---

## Related documents

- [Platform Proof Protocol](PLATFORM_PROOF_PROTOCOL.md)
- [Platform Proof Map](PLATFORM_PROOF_MAP.md)
- [Platform proofs README](README.md)
- [Public proof dashboard](../docs/project/proofs/PROOFS.md)
- [Public proof and claims model](../docs/project/maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md)
