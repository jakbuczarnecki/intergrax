# INTEGRAx-NPSC-5F-R3-FINAL-REQUALIFICATION

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-NPSC-5F-R3-FINAL-REQUALIFICATION` |
| **Branch** | `development` |
| **Implementation / re-freeze SHA** | `aa3b43456a530e1e2f50b81cab486874fe06e3b1` |
| **Global frozen platform baseline** | `a185403d0c7524c29bea2fe09212f9508e6bccd8` (**unchanged**) |
| **Prior NPSC-5F Evidence Plane baseline (EE-FINAL-02)** | `7a3569c64e892588992635c9cee10c264a9fc200` |
| **Prior NPSC-5F/R3 Final implementation** | `0346face3ef68d8f21504822a26f8f45f2384cf9` |

## Requalification Scope

Scoped requalification and protected-drift resignoff for:

- NPSC-5F R3 export boundary (`export_boundary.py`, governed causal export versioning)
- NPSC-5F Final Evidence Plane (causal evidence v1/v2, persistence decode, background identity compatibility)
- `RuntimeExecutionRef`, `PlatformCausalEvidence`, `CausalEvidenceExportSource`, background execution identity v1/v2
- Compatibility / dual-read / enrichment / conflict hardening (`962bf1ade` → `aa3b434`)

Out of scope: global platform baseline, ExecutionRuntime, Governance, Retry, Recovery, Nexus, EE-B2/B3/B4.

## Architecture Reopen Chain

```text
962bf1ade → 1858ba068 (Class C) → fe2b1510 (scoped reopen APPROVED)
→ 91f6c864 (compatibility ACCEPTED) → aa3b434 (persistence hardening PASS)
```

No additional production semantic drift in scoped surfaces between `aa3b434` and requalification HEAD.

## Accepted Class C Drift

Production files requalified (from `962bf1ade`, `91f6c864`, `aa3b434`):

| Area | Paths |
| ---- | ----- |
| Causal evidence | `causal_evidence.py`, `causal_evidence_legacy.py`, `platform_causal_evidence_codec.py`, `causal_evidence_record_codec.py`, `causal_evidence_enrichment.py`, `document_store_causal_evidence_persistence.py`, `persistence_conformance.py` |
| Export | `export_boundary.py`, `causal_evidence_export.py` |
| Background identity | `identity_types.py`, `identity_record_codec.py`, `identity_dual_read.py`, `identity_persistence.py`, `bootstrap.py`, `reentry_admission.py`, `required_audit_evidence.py` |
| Contracts | `intergrax/contracts/npsc5f_compatibility.py` |

Not absorbed: EE-B2 chaos commits, execution capacity/reliability contract churn, unrelated EE-B3-A WIP.

## Contract Semantics

- `platform_causal_evidence.v1` — frozen legacy **read**
- `platform_causal_evidence.v2` — canonical **write**
- Read v1 + v2; write v2 only (`ForbiddenPlatformCausalEvidenceV1WriteError`)

## Compatibility Semantics

- Partition-aware background identity decode (v1 triplet / v2 quadruplet)
- Fail-closed conflicts; no heuristic precedence (no “prefer v2”, “prefer non-null”, “prefer first”)
- v1 + v2 matching triplet → PASS; task/run/attempt mismatch → typed fail

## Identity Authority

- Execution identity authority = canonical mint owner (`DefaultExecutionIdentityAuthority`)
- Observability enrichment does not mint; migration/dual-read does not mint
- `CanonicalExecutionIdLookupPort`: bounded, tenant-scoped, exactly-one-match, no minting

## Evidence/Control

No v1/v2 path authorizes execution, retry, recovery, scheduler, or Governance mutation.

## Persistence

Platform semantics → persistence port → store; NPSC-5F core without vendor branching. KV / DocumentStore / causal evidence share codec + reconciliation semantics.

## Export R3

- `causal_evidence_export_source.v1` — no `target_execution_id` (explicit legacy)
- `causal_evidence_export_source.v2` — required `target_execution_id`
- Explicit export version selection; no implicit v2→v1 downgrade; canonical production default **v2**

## Final Evidence Plane

Evidence plane sentinel baseline advanced to implementation SHA `aa3b434…` (scoped qualified baseline, distinct from global `a185403d…`).

## Regression Matrix

| Gate | Result |
| ---- | ------ |
| P0 reconciliation | PASS |
| R1 durable boundary + Final sentinel | PASS (excluding drift until resignoff) |
| R2 Final | PASS (core + sentinel; no R2 baseline change) |
| R4 impacted regression | PASS (mandatory matrix, no R4 baseline change) |
| R3 full qualification | PASS (post resignoff) |
| Final full qualification | PASS (post resignoff) |
| Identity (EE-A2-H3, compatibility, conflict hardening) | PASS |
| Final mandatory regression matrix | PASS (518+ tests, post resignoff) |

Session logs: `.tmp/session/npsc5f-r3-final-requal/`.

## Protected Drift Before

| Protected family | Old baseline | Drift files | Qualified? |
| ---------------- | ------------ | ----------- | ---------- |
| R3 export | `0346face…` | `export_boundary.py` | Yes — Class C export v2 |
| Final Evidence Plane | `7a3569c64…` | 9 observability causal/export/codec paths (see classifier run) | Yes — scoped reopen chain |

## Baseline/Fingerprint Advancement

| Symbol | Old | New |
| ------ | --- | --- |
| `R3_IMPLEMENTATION_SHA` | `0346face…` | `aa3b434…` |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `7a3569c64…` | `aa3b434…` |
| `NPSC_5F_R3_FINAL_SHA` | `0346face…` | `aa3b434…` |

R1/R2/R4 post-qualified baselines unchanged.

## Protected Drift After

Sentinels expect **empty** drift from `aa3b434…` to `origin/development` on R3 protected paths and **empty** BREAKING Evidence Plane drift from scoped baseline.

## Scoped Re-freeze

- **NPSC-5F R3** = REQUALIFIED / RE-FROZEN @ `aa3b434…`
- **NPSC-5F FINAL** = REQUALIFIED / RE-FROZEN @ `aa3b434…` (scoped Evidence Plane baseline)

## Architecture Reopen Closure

Scoped Architecture Reopen (`fe2b1510…`) = **CLOSED**  
Class C change = **requalified**

## Tests

Targeted: `test_npsc5f_v1_v2_compatibility.py`, `test_identity_persistence_conflict_hardening.py`, R3/R3 Final, Final qualification, `test_npsc5f_final_protected_drift.py`, `test_npsc5f_r3_protected_drift.py`, mandatory regression matrix.

## Static Quality

`ruff check/format --check` on changed Python helpers/tests; `git diff --check` before commit.

## Production Changes

**NONE** in this requalification commit (baseline / qualification record only).

## Findings

| Severity | ID | Note |
| -------- | -- | ---- |
| — | — | No blocking production defects in scoped surfaces |

## Final Verdict

**NPSC-5F R3 + FINAL REQUALIFICATION = PASS**
