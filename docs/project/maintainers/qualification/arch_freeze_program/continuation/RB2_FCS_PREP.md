# RB-2 / FCS — operator-signature package (draft only)

**Banner:** Prep artefact for the Operator or a recorded delegate. **Not** an approval, **not** RB-2 activation, **not** FCS, **not** G2, **not** M3, **not** ARCH-FREEZE CLOSED.

**Worktree:** `D:/Projekty/_wt/intergrax/af-fc-rb2` · branch `af-fc/rb2-prep`  
**Ticket (commit trailer):** `T-IGX-050-N` (operator-signature prep only; does not execute PT-Q03).

## Register rows (read-only cite)

| Row | MEP milestone | Register state | Effect for this package |
|-----|---------------|----------------|-------------------------|
| **OD-AF-12** | M4 — RB-2 activation line at G2 | **OPEN** — not yet decided; to be written by the Operator or a recorded delegate | Draft activation line below; **not signed** |
| **OD-AF-22** | Packet-3 train approvals (DECIDED) | Includes: the merge of the train into `development` (**FCS**) **stays the Operator's decision** | Draft FCS merge description below; **not executed** |

Sources: [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md); [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) (M4 / M5 decision list; §L RB-2 field map; ladder rung `FCS`; node `T-AF2-RW-MERGE`).

## Candidate train tip (not G2, not FCS)

| Field | Value |
|-------|-------|
| Branch | `arch-freeze/fc-train` (worktree application branch when cut) |
| **Candidate tip SHA** | `ea923cf03d88fc9d22637796e9002eb9d68e6e59` |
| Role | **Candidate** RW / fc-train head for RB-2 prep only — **not** recorded as **G2**, **not** recorded as **FCS** |

Future **G2** is the wave-audited SHA once PT-Q02 and train closure criteria are met (`MEP-AF2_v2.1` ladder). This pin is cited only to anchor the prep package.

---

## OD-AF-12 — RB-2 activation line (draft)

**Decision status:** **UNSIGNED**

Register: OD-AF-12 remains **OPEN** until the Operator or a recorded delegate authorizes RB-2 activation at **G2**. No delegation or closure is claimed in this file.

**Operator decision file** (future path pattern): `docs/project/maintainers/qualification/IIIC_OD_ACTIVATION_ARCH_FREEZE_RB2_*.md` or successor named in PT-Q03 handoff.

**Required activation line** (`MEP-AF2_v2.1` §L — operator decision file, `iiac_activation_gate.py` grammar):

```text
Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>
```

| Subfield | Prep value |
|----------|------------|
| Policy version | `IIAC-2026-11-05.1` |
| Line form | Exact regex line above; `<timestamp>` = ISO-8601 at signature time |
| `Decision status` (companion file) | **DRAFT** until Operator/delegate signs — target **AUTHORIZED** only after mandate |
| `audited_implementation_sha` / G2 binding | **Unset in this package** — binds at **G2**, not at candidate tip |

**Signature slot (placeholder — do not treat as Operator approval):**

```text
Approved IIAC policy activation IIAC-2026-11-05.1: operator @ ________________
```

(Correct policy id in production line: `IIAC-2026-11-05.1` per MEP §L; placeholder row retained for handoff editing.)

---

## OD-AF-22 / FCS — merge into `development` (draft)

**Merge status:** **NOT_MERGED**

OD-AF-22 (**DECIDED**): train ticket approval packets and the signing delegate handle `Approved:` lines on the train; **the merge of the train into `development` (FCS) stays the Operator's decision**. This package does **not** record that decision as taken.

**FCS merge description** (`MEP-AF2_v2.1` ladder rung `FCS`, node **T-AF2-RW-MERGE**, §N2 product-tree rule):

| Item | Specification |
|------|----------------|
| Source branch | `arch-freeze/fc-train` |
| Target branch | `development` |
| Commit kind | **Merge commit** (not fast-forward-only product rewrite) |
| Product-path constraint | `intergrax/`, `agents/`, `applications/` tree at **FCS** must be **identical** to the **future G2** pin (same trees as at FCS ≡ G2 per ladder) |
| Non-product | Docs-only delta allowed on rung **R** (`git diff G2 R` docs only); FCS itself is the train→`development` merge |
| Operator gate | Explicit Operator decision required per OD-AF-22 before merge is performed |
| IIAC | `verify_iiac_policy_activation.py --require-in-force` exit 0 at FCS is a separate W4 gate (`MEP-AF2_v2.1` W4 table) — **not claimed here** |

**Draft merge record (unexecuted):**

```text
FCS: merge arch-freeze/fc-train → development @ merge commit <TBD>
Product tree @ FCS ≡ G2 @ <future G2 SHA TBD>
Candidate train tip (prep cite only): ea923cf03d88fc9d22637796e9002eb9d68e6e59
```

---

## Package verdict

| Layer | Verdict |
|-------|---------|
| Operator-signature prep | **DRAFT** — lines packaged for future signing |
| OD-AF-12 activation | **UNSIGNED** |
| FCS merge | **NOT_MERGED** |
| G2 / FCS / M3 ladder | **Not set** by this artefact |

No agent may sign any decision. OD-AF-AUTO-05: no autonomous FCS, G2, or CLOSED claims.