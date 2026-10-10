# TRACE-X-P5-R2-P4 — Configured / Effective Reconstruction Projection

| Field | Value |
|---|---|
| **Parent** | **TRACE-X-P5-R2-P4** |
| **Current disposition** | **CLOSED / independently accepted** |
| **Corrected implementation chain** | **TRACE-X-P5-R2-P4-R2** — qualification [`TRACE_X_P5_R2_P4_R2_CORRECTED_RECONSTRUCTION_REQUIREMENT_EVIDENCE_RECOVERY.md`](TRACE_X_P5_R2_P4_R2_CORRECTED_RECONSTRUCTION_REQUIREMENT_EVIDENCE_RECOVERY.md) |
| **Accepted SHAs** | blocker **32** @ `f21ffbae04b12686784933ef6926764198c19201`; blockers **33**/**35** @ `90bfa980d2f6d445f6300fce729247e5400de648`; blocker **34** final correction @ `25df09093bf4d7103f0bf66780ae794d52a7f929` |
| **Blockers 24–35 (P4-R2 scope)** | **0** unresolved |
| **REJECTED HISTORICAL BASELINE** | `055ed448cb890026c8c34e336e7baf7422daa2f6` — preserved as lineage only; **not** current status |
| **FRZ-TRC-11** | **OPEN** (remaining ownership = P5 closed-world qualification + **TRACE-X-CERT**) |
| **P5 closed-world / TRACE-X-CERT** | **NEXT / NOT ENTERED** |
| **Tenant Isolation Audit (this reconciliation)** | **N/A — WITH EVIDENCE** (no runtime tenant mechanism changes; P4-local tenant evidence preserved) |

## Current evidence (accepted corrected P4-R2 chain)

- **P3** + **P4** parent closure: configured execution → durable **PinRecord** → mandatory requirement spine → reconstructable historical configured/effective provenance.
- **Durable Case C/D** accepted (blocker **33**).
- **Option B as-of** + requirement spine design locks **24–31** accepted in qualification lineage.
- **Historical configuration mutation** does not alter historical truth.
- **`P4_INTEGRITY_QUALIFICATION_MATRIX` = 21/21** PASS (blocker **34** @ **R1-R1-R1**).
- **Active staging tenant continuity** accepted (blocker **35**).
- **Tenant-local P4 audit** = **PASS**; **global FRZ-TEN** unchanged (**TENANT-X**).

R1 reconciliation lineage (authority / emission / retry / recovery locks): [`TRACE_X_P5_R2_P4_R1_RECONSTRUCTION_REQUIREMENT_AUTHORITY_AS_OF_RECONCILIATION.md`](TRACE_X_P5_R2_P4_R1_RECONSTRUCTION_REQUIREMENT_AUTHORITY_AS_OF_RECONCILIATION.md) through [`TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_R1_AMBIGUOUS_PIN_OUTCOME_STAGING_TIMESTAMP_SEMANTICS.md`](TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_R1_AMBIGUOUS_PIN_OUTCOME_STAGING_TIMESTAMP_SEMANTICS.md) — **historical design lineage**; closure evidence = **P4-R2** qualification artifact above.

---

## Historical — rejected initial implementation baseline (`055ed448…`)

**Status @ rejection:** **BLOCKED** (superseded by **P4-R2**).

**Rejected implementation baseline:** `055ed448cb890026c8c34e336e7baf7422daa2f6` (**REJECTED HISTORICAL BASELINE**)

### Blockers (independent audit @ rejected baseline)

| ID | Summary |
|---|---|
| `R2-P4-PROVENANCE-REQUIREMENT-AUTHORITY-24` | Magic runtime payload requirement — no production emitter; fail-closed not mechanical |
| `R2-P4-AS-OF-CONFIG-PROVENANCE-FUTURE-LEAK-25` | `read_all` against non-temporal P2 store under `execution_as_of` |
| `R2-P4-REQUIREMENT-EVIDENCE-EMITTER-BOUNDARY-26` | Integrations must not own runtime spine emit — P4-R1-R1 |
| `R2-P4-PIN-REQUIREMENT-EVIDENCE-DUAL-WRITE-27` | Pin vs spine dual-write protocol — P4-R1-R1 |

Resolved in corrected **P4-R2** chain (blockers **24–35** = **0** unresolved).

### Inventory @ rejected baseline (lineage only)

| Artifact | Path |
|---|---|
| Neutral reader Protocol | `intergrax/contracts/execution_integration_configuration_provenance.py` |
| Pinning-store reader adapter | `intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py` |
| Reconstruction projection | `intergrax/runtime/observability/reconstruction/integration_configuration_provenance_projection.py` |
| Reconstructor injection | `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` |
| Reconstruction DTO fields | `intergrax/contracts/execution_reconstruction_models.py` |
| Diagnostic composition | `intergrax/applications/_shared/diagnostic_composition.py`, `diagnostic_read_wiring.py` |
| Host wiring | `harness_host_runtime.py`, `scenario_runtime_baseline.py` |

### Superseded claims (rejected baseline)

- Runtime payload `execution_integration_configuration_provenance_required` as requirement authority — **rejected** in P4-R1; superseded by durable requirement spine in **P4-R2**.
- “Unresolved / ADR: None” — **incorrect** @ rejected baseline; remediated in **P4-R1**/**P4-R2** lineage.

### Tests (rejected baseline — not current audit evidence)

- Unit: `tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py`
- Gates: `tests/qualification/trace_x/test_trace_x_p5_r2_p4_reconstruction_gates.py`

Current audit evidence: **P4-R2** qualification + integrity matrix + durable Case C/D proofs (see linked qualification artifact).
