# RED_FIRST — G2 disposition (strict evidence @ fc-train)

**Authority:** Grok orchestrator binding decision. **Not** Operator ratification.  
**Evidence pin:** `3991186b1` on branch `af-fc/red-strict` (parent `a678671fe`, integration head at strict replay).  
**Train application:** `arch-freeze/fc-train` fast-forwarded to include strict artifacts at `3991186b1` before this disposition commit.  
**Strict artifacts:** `RED_FIRST_STRICT_RESULT.md`, `RED_FIRST_STRICT_RESULT.json` (same directory).

## Binding dispositions

### RED_AT_PREFIX (9 rows — red-first evidence)

These rows measured **RED_AT_PREFIX** under strict replay at the evidence pin. They **count as red-first evidence** for ARCH-FREEZE qualification traceability. They are **not** rewritten to any other measurement class.

| `frz_id` | measurement |
| --- | --- |
| `FRZ-BND-04` | **RED_AT_PREFIX** |
| `FRZ-BND-05` | **RED_AT_PREFIX** |
| `FRZ-CFG-06` | **RED_AT_PREFIX** |
| `FRZ-CTR-02` | **RED_AT_PREFIX** |
| `FRZ-CTR-06` | **RED_AT_PREFIX** |
| `FRZ-OWN-05` | **RED_AT_PREFIX** |
| `FRZ-TYP-01` | **RED_AT_PREFIX** |
| `FRZ-TYP-03` | **RED_AT_PREFIX** |
| `FRZ-TYP-05` | **RED_AT_PREFIX** |

### RED_FIRST_UNREPRODUCIBLE (8 rows — closed for G2/SCOPE blocking)

These rows measured **RED_FIRST_UNREPRODUCIBLE** under strict replay at the evidence pin.

| `frz_id` | measurement |
| --- | --- |
| `FRZ-CTR-01` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-PLG-01` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-PLG-02` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-PLG-07` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-TEN-01` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-TEN-02` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-TEN-07` | **RED_FIRST_UNREPRODUCIBLE** |
| `FRZ-TEN-10` | **RED_FIRST_UNREPRODUCIBLE** |

**Orchestrator rules (binding):**

1. **No reclassification** — none of the eight rows are rewritten as **RED_AT_PREFIX** or any other strict class.
2. **Not open product defects** — integrated negative tests for these rows exited **0** at integration head `a678671fe` (see strict result `integrated ok` column).
3. **Non-blocking** — the eight **RED_FIRST_UNREPRODUCIBLE** rows do **not** block **G2** gate measurement or **SCOPE** measurement for ARCH-FREEZE continuation.

## Explicit non-effects

This disposition does **not**:

- set **G2** closed or satisfied;
- set **FCS** closed or satisfied;
- set **M3** closed or satisfied;
- declare **ARCH-FREEZE CLOSED**.

**Recorded:** 2026-11-25 (fc-train, post–`3991186b1` evidence import).