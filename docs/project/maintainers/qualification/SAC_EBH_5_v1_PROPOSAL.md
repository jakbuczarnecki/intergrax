# SAC-EBH-5-v1 — Stage Acceptance Contract (Proposal)

| Field | Value |
| --- | --- |
| **Contract ID** | `SAC-EBH-5-v1` |
| **Status** | **PROPOSED / NOT FROZEN** |
| **Freeze gate** | `OPERATOR APPROVED — EBH-5 SAC FREEZE` |
| **Prerequisite** | DIST-SCALE-X = **CLOSED — OPERATOR SCOPED ACCEPTANCE** |

## Frozen acceptance (when approved)

1. Mechanical exit criteria roadmap §3.0.8 (inventory, matrix, reachability, behavioral proofs, fixed substrates, SQLite annex, RPL-G).
2. RPL-A..H per [VENDOR_BACKEND spec](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md).
3. CONFIG-X PASS @ `fac9a43f…` preserved; current-HEAD obligations additive only.
4. EBH-6/DUP-X boundaries: no universal hardcoding register in EBH-5 scope.
5. Test budget: L2 consolidated `tests/qualification/ebh_5/`; L4 physical/vendor only with OD-EBH5-03.
6. Exact-SHA provenance on all dossier artifacts.
7. Independent Protocol 03 mandatory for parent CLOSED.

## Freeze readiness checklist (implementation side)

| Item | Ready |
| --- | --- |
| SSOT inventory/matrix/proof registry | Yes @ CERTIFICATION_HANDOFF_SHA |
| Five QUALIFIED behavioral proofs resolvable | Yes |
| Protocol 03 handoff dossier | Yes |
| RPL-G + CONFIG-X blockers empty | Yes (gates) |
| Independent AUDIT PASS | **No** — OD-EBH5-07 |
| Operator SAC freeze recorded | **No** — OD-EBH5-02 |

## Rozbieżności / open points (not SAC violations until auditor rules)

- **51 NOT YET QUALIFIED** — intentional; EBH-7 convergence.
- **RPL-F NA** on QUALIFIED rows — no failure/recovery L3 in this stage.
- **OTLP empty endpoint** vs **DISTRIBUTED_OTLP ConfigurationError** asymmetry — documented in final correction dossier.
- **Reachability** skip when graph integration module absent.
- **Historical SHA ladder** — multiple commits; auditor must use CERTIFICATION_HANDOFF_SHA only for verdict.

## Do podpisu operatora (not executed by agent)

```text
OPERATOR APPROVED — EBH-5 SAC FREEZE
```

Record in [`EBH_5_OPERATOR_DECISION_REGISTER.md`](EBH_5_OPERATOR_DECISION_REGISTER.md) OD-EBH5-02 only after independent review of this proposal.

## Operator decisions required

See [EBH_5_OPERATOR_DECISION_REGISTER.md](EBH_5_OPERATOR_DECISION_REGISTER.md).
