# EVOL-X E2 — Cause oracle preflight record

**Finding:** `F-EVOLX-E2-CAUSE-PREFLIGHT`  
**Disposition:** **`EVOL-X E2 CAUSE PREFLIGHT — BLOCKED / EXACT QUALIFICATION DEFECT`**  
**Audit HEAD:** `5333c13f1cb9c55e10c3fc4929a1391eee2dac22`  
**TES:** `dcd7ae6e264d435994c61162e0d41aba25f71283` · **TECH:** `9f875fc62d1f6b2231b31c843ae0428a2d871031` (unchanged)

**Session log:** `.tmp/session/evol-x-e2-m09h/preflight-cause-oracle.log`

---

## Verdict summary

| Question | Answer |
|----------|--------|
| Incorrect reconstruction formula only? | **No** — exhaustive search (impact nodes, dossier, workflow A7, path/input composites) @ `cb407403`, `6fec9ec`, `dcd7ae6` does not reproduce registry hashes |
| Legitimate undocumented transform? | **No** — no normative doc or code defines `expected_cause_identities` derivation |
| Frozen registry oracle defective? | **Yes** |
| Diagnostic packet vs verifier demand? | **Yes** — packet exposes `node_id` + impact; Mode A requires opaque 64-hex values that are **not** `stable_digest_hex([kind, node_id])` for the seed DIRECT node |

**Oracle correctness:** **FAIL**  
**Evaluator solvability (independent PASS):** **FAIL** — honest derivation from allowed evidence cannot satisfy Mode A.

---

## C1–C3 discrepancy matrix (@ active TES)

Qualified diagnostic path: `build_closed_world_inventory` → `assess_path_change_impact(changed_paths={seed})` (same as E2 operator tool).  
**Canonical candidate identity** (only graph-stable formula implied by contracts; **not** written in M-09H spec): `stable_digest_hex([kind, node_id])` for the **DIRECT** affected node of the seed module (`testing_support/change_safety/contracts/canonical_json.py`).

| Case | Seed path | input SHA256 @ TES | DIRECT `node_id` | Derived `stable_digest_hex([kind, node_id])` | Registry `expected_cause_identities[0]` | Match |
|------|-----------|-------------------|------------------|-----------------------------------------------|----------------------------------------|-------|
| C1 | `intergrax/contracts/execution_identity.py` | OK (`91f6d530…`) | `module:intergrax.contracts.execution_identity` | `3fc41379bf4835887ef1758cc1aedf2f966e2280cb58f9b14da09bbfa48b091d` | `9328b3d65d4e54c1ccfa9298d440915c5f7de2b15cbc557148fb518ba4bf25ed` | **NO** |
| C2 | `testing_support/change_safety/graph/impact.py` | OK (`dda029e1…`) | `module:testing_support.change_safety.graph.impact` | `f1decda53e2306a25418e70ed30046d491f36639e30cb0bfb52be7db1d61f83e` | `a2e34afb13fa8f11947bde48275e209909f88923b3f4c51b5c6676bb8b48c112` | **NO** |
| C3 | `testing_support/change_safety/contracts/enums.py` | OK (`1feef84e…`) | `module:testing_support.change_safety.contracts.enums` | `82364099afe3fa41f42a9f0860ce54bebf4c53ec9f740f04859277ac6069ca55` | `b41609cafc86c65456508ab2ac5defbc45159e7014aebd6793d730f9780ff97b` | **NO** |

Historical note: registry introduced @ `6fec9ec387089dfcee1039089ba5ad901fab39d6` with different expected hashes (`23c536f6…`, `fddf2cf6…`, `ee4031eb…`) — also **non-reproducible** @ bound `cb407403`. Bundle republish `43a44b492` swapped expected values to current triple **without** a documented derivation or code change — evidence-only edit.

---

## Causal identity formula

**Explicit absence in product contract:** `M09DrillCaseRegistryEntry.expected_cause_identities` is validated for shape only (`testing_support/change_safety/contracts/artifacts.py`). P03 Mode A compares `attributed_cause_identities` to registry bytes only (`testing_support/change_safety/evidence/p03.py`) — **no** check that registry matches diagnostic tooling.

**Implied but unpublished candidate:** `stable_digest_hex([DIRECT|kind, node_id])` for seed module DIRECT node (see matrix). Not normative; **does not match** frozen registry anyway.

---

## Qualification defect — remediation boundary (not executed here)

**Label:** `F-EVOLX-E2-CAUSE-PREFLIGHT — CONFIRMED QUALIFICATION DEFECT`

| Area | Required corrective work (separate authorized implementation) |
|------|------------------------------------------------------------------|
| Normative | Define M-09H cause identity in metric spec + runbook (single deterministic function of diagnostic output @ TES) |
| Registry | Recompute `expected_cause_identities` for C1–C3 @ active TES using that function; publish only in **EVIDENCE_FINAL** rebound registry (do not silently patch TECH history) |
| Gates | VL3 test: registry expected == `compute_m09_cause_identity(seed, inventory @ TES)` fail-closed |
| Optional code | Shared helper used by operator tool + verifier pre-check (would imply **new TES** + CI re-qualification) |
| **This audit** | **No** change to qualified code, tests, TES, TECH, P03 bundle, or registry oracle bytes |

**TES/TECH invalidation impact:** Documentation-only preflight → **none**. Registry-oracle fix in evidence → FINAL supplement only if identity rule is frozen without code change; **any** new derivation helper in `testing_support/` → **new TEST_EXECUTION_SHA** + W6-04 re-run per SHA ladder.

---

## E2 authorization

**Human M-09H drill:** **NOT AUTHORIZED** until registry oracle and documented derivation align.

**Next E2 action:** Operator scopes **E-CORR-M09H-ORACLE** implementation batch (formula + registry recompute + gate); then re-run this preflight → target disposition `PASS / HUMAN DRILL AUTHORIZED`.

---

## Auditor-only oracle confidentiality

Expected 64-hex values remain in [AUDITOR_ONLY/EVOL_X_E2_ORACLE_ANSWER_KEY.md](./AUDITOR_ONLY/EVOL_X_E2_ORACLE_ANSWER_KEY.md) for Mode A reference only; preflight confirms they must **not** be used as human-facing targets until defect closure.
