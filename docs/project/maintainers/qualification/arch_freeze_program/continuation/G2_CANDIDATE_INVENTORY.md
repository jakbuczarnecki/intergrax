# G2 candidate inventory (describe only — no pin)

**Document role:** Inventory of SHA lineage and enterprise gates relevant to a **future** G2 candidacy. **Does not pin G2**, **does not** declare G2 satisfied, and **does not** close **SCOPE**, **FCS**, **M3**, **ENTRY**, or **ARCH-FREEZE**.

## What G2 is (expectation, not a decision)

Future **G2** must be a **new** `arch-freeze/rw-train` tip **after** the published train:

| Role | SHA |
| --- | --- |
| Published train tip (evidence base, **not G2**) | `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a` (`origin/arch-freeze/rw-train`) |

That successor tip must carry **RW product drift** observable relative to red-first / integration evidence (MEP-AF2 ladder: RW merges landing on train, then publication as `af2/pin-<sha>`). This file **describes** the ladder; it does **not** name a pin SHA or create a branch.

## SHAs that are explicitly **not** G2

| SHA | Role |
| --- | --- |
| `2a6e197ca47981deca2bf35b8159c4e684be535b` | `development` integration context for V5 residual docs (**not G2**) |
| `e32f82890` | Cited development-line artifact; **not G2** |
| `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a` | Published **rw-train** tip — G2 **base lineage only**, not G2 itself |

## Publication evidence SHAs (V5 packages — evidence only)

| Package | Merge / publication SHA | Notes |
| --- | --- | --- |
| P1 (red-first V5 integration) | `2411e4118` | Evidence of P1 merge; does not pin G2 |
| P2 (Campaign B V5 alpha dossier) | `9a5b796d9` | Evidence of P2 merge; does not pin G2 |
| Residual docs base | `2a6e197ca` | Current `development` head for this doc pass; **not G2** |

These SHAs support audit traceability only. They do **not** substitute for a post-`e9bebe0ac` train tip plus G2 checklist / PT-Q02 governance.

## Enterprise parallel — EC-V7-A (`agent-env/plans/ec-v7-a-product-completion.md`)

| Field | State |
| --- | --- |
| Plan verdict (default) | **`EC-V7-A_PARTIAL_EXACT_BLOCKERS`** |
| A01 Seam Ledger | **`BLOCKED_BY_EXACT_AUTHORITY`** — no merge into factory candidate without Approved write-set |
| A02 Closed-World | **`BLOCKED_BY_EXACT_AUTHORITY`** — same |
| Train policy | **Do not merge `arch-freeze/rw-train`** into EC-V7-A factory candidate |
| EC-02 | **Do not declare EC-02 CLOSED** or product-complete from this inventory |

EC-V7-A runs on `development` / factory branches; it does **not** satisfy or replace ARCH-FREEZE G2.

## W3 product-edit gate

While **SCOPE** remains **OPEN**, **W3 product edits stay forbidden** on the ARCH-FREEZE ladder (MEP-AF2). This inventory does not waive W3.

## Related continuation pointers

- Train CI / replay context: [`TRAIN_CI_REPLAY_PLAYBOOK.md`](TRAIN_CI_REPLAY_PLAYBOOK.md) @ `e9bebe0ac`
- MEP G2 candidacy wording: [`../MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) (expectation: new tip after `e9bebe0ac`, then `af2/pin-<sha>`)
- Draft checklist (not authoritative pin): [`G2_CHECKLIST_DRAFT.md`](G2_CHECKLIST_DRAFT.md)

## Explicit non-effects

- **No G2 pin** and **no G2 CLOSED** claim.
- **No** merge of `rw-train` into enterprise factory candidate.
- **No** change to **SCOPE**, **R**, **FCS**, **M3**, **ENTRY**, or operator **DECIDED** cards.

**Recorded:** 2026-11-26 · G2 candidate inventory (`af-v5/residual-docs`).