# DUP-X — Protocol 03 Independent Audit Pass Record

| Field | Value |
| --- | --- |
| **Record type** | ECAP §2.B **B6** — Independent exact-SHA acceptance |
| **Status** | **ISSUED — PASS (SCOPED)** |
| **Verdict** | **PASS (SCOPED)** — Protocol DUP-X-03 **B1–B6** |
| **INDEPENDENT_ACCEPTANCE_BASELINE_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **QUALIFICATION_EXECUTION_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **PRE_ACCEPTANCE_PUBLICATION_HEAD** | `1f782aaa9017b31b2e68a1490d186a1e3fa055cd` (doc-only delta after qual pin — reviewed) |
| **Auditor identity** | **ChatGPT** — external evidence-based Protocol DUP-X-03 review (**not** Cursor Agent) |
| **Publication** | Repository commit publishing this record — implementer-controlled docs only; provenance preserved above |
| **Date** | **2026-10-29** |
| **Evidence bundle** | `.tmp/proof/dup-x-b6-independent-evidence-bundle/` |

## Scoped acceptance (explicit)

PASS applies to **DUP-X Cross-Architecture Duplicate & Bypass Certification** @ **`INDEPENDENT_ACCEPTANCE_BASELINE_SHA`** only. **Not** global **FRZ PASS** · **not** **PROD-Q** · **not** **ARCH-FREEZE** · **not** automatic **ROADMAP-REPLAY-X** closure · **not** parent **`CLOSED`** without operator **OD-DUPX-08**.

## B5 replay attestation (auditor disclosure)

| Method | Disposition |
| --- | --- |
| Independent pytest re-execution @ baseline | **Not performed** by external auditor |
| Log/evidence inspection | **Performed** — `dup_x-full.log` (**44 passed**), `l3-parent.log` (**24 passed**), `b4-negative-gates.log` (**4 passed**), NPSC allowlist log (**1 failed**, **10** modules) |
| ECAP additional replay | **Formally not required** for this scoped PASS given inspected implementer bundle + SHA integrity @ baseline |

## B1–B5 disposition

| Block | Focus | Disposition | Notes |
| --- | --- | --- | --- |
| **B1** | Scope · 15 families · CAND-01..15 | **PASS** | Closed-world inventory + class matrix |
| **B2** | SHA ladder @ baseline | **PASS** | C0-04..07 · manifest; qual pin `a7d83925b…` |
| **B3** | Factory-first · deltas · Class D | **PASS** | Responsibility delta dossiers + qual SSOT |
| **B4** | COMPAT-R3 · NPSC ten-module · negative gates | **PASS (SCOPED)** | B4 negative **4 passed**; global NPSC allowlist **still FAIL** — see exclusion |
| **B5** | Full `tests/qualification/dup_x/` | **PASS** | **44 passed** @ `QUALIFICATION_EXECUTION_SHA` (log-inspected) |

## B6 verdict

| Field | Value |
| --- | --- |
| **Verdict** | **PASS (SCOPED)** |
| **Date** | **2026-10-29** |
| **Findings** | None blocking scoped acceptance |
| **NPSC-4.2 exclusion** | **ACCEPT** for DUP-X only — [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md); allowlist gate remains **FAIL**; no allowlist weakening |

## Parent closure (operator only)

**OD-DUPX-08** remains **OPEN** until operator records:

`OPERATOR APPROVED — DUP-X CLOSED (OD-DUPX-08)`

Until then platform status:

**`DUP-X — INDEPENDENTLY ACCEPTED / OPERATOR CLOSURE PENDING`**

## Forbidden interpretations

- Cursor Agent is **not** the independent auditor for B6.
- This record does **not** authorize **ROADMAP-REPLAY-X** entry.
- **new global FRZ PASS = 0**
