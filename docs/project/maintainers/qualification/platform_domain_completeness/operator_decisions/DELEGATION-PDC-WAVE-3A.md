# DELEGATION-PDC-WAVE-3A — Operator commission for plan-approval signing (wave 3a product batch B-D6)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-3A |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-1.md` and `DELEGATION-PDC-WAVE-2.md` |
| **Date** | 2026-11-12 |

## Commission (transcription)

1. The Operator held B-D6 until the end of wave 2, in line with the dependency DAG (`PDC_DEPENDENCY_DAG.md`: B-D2 and B-D5 precede B-D6). Chat answer, English rendering: **"D6 will wait for the end of wave 2"** (Polish original: "D6 poczeka na koniec Fali 2").
2. Wave 2 is implemented (draft PRs #489 for B-D2 and #488 for B-D5, not merged). The Operator then answered in chat, in English rendering: **"Start D6"** (original wording in Polish: "Start D6").
3. The Operator's earlier decisions still bind this wave: product code is implemented in isolated worktrees and merged only after MCP-SERVER-X P8 is published (OD-PDC-SEQ-01), and the `ollama` regression (B-REG-P) is excluded.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-P** only.
- The delegate writes these lines only after confirming all of the following:
  - the plan digest matches;
  - the plan matches batch B-D6 in `PDC_IMPLEMENTATION_WBS.md` (the write set equals the WBS write set, every requirement row is covered) and the Accepted ADR-PDC-06 with OD-PDC-SCOPE-07 = A;
  - the write set stays inside the batch's own lane and writes under the shared hotspot `intergrax/applications/_shared/` only the four files named in the WBS;
  - the plan creates no new policy engine, identity, trust or audit authority beside the existing owner;
  - the plan does not weaken any governance invariant and states that the PR is not merged before MCP-SERVER-X P8.
- **Not delegated:**
  - OD signatures;
  - IIAC activation (PDC-P7-00);
  - merges, including any merge before MCP-SERVER-X P8;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - tickets M, N, O, Q..S and any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement T-IGX-036-P.

## Amendment 1 — write set of B-D6 (T-IGX-036-P)

- **Operator decision** (chat, English rendering of the Polish original "Tak, rozszerz"): **"Yes, extend"**, in answer to the question whether the write set of B-D6 may be extended to close PDC-D6-R027 and the dependencies found by the first implementation (draft PR #494).
- **Extension** (eight additional paths): `intergrax/runtime/attestation/host_attestation.py`, `intergrax/runtime/attestation/__init__.py`, `intergrax/applications/_shared/attestation_runtime_bridge.py`, `intergrax/core/plugins/discovery.py`, `intergrax/runtime/nexus/tools/plan_context_invocation.py`, `scripts/maintenance/check_critical_action_signing.py`, `scripts/maintenance/check_immutable_security_audit_trail.py`, `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` (cases 4_1 and 23_1 only). Gates are updated to assert the new fail-closed behavior; none is weakened or removed.
- **Hotspot:** `attestation_runtime_bridge.py` is a fifth file under `intergrax/applications/_shared/`; the delegate's earlier check "only the four WBS-named files" is replaced by "only the five files named in the amended WBS". B-D6 is the only wave 3a batch, and `pdc_x_wbs.py --check` reports no write conflicts.
- **Effect on the signature:** the plan digest of T-IGX-036-P changed, so the earlier `Approved:` line no longer applies. The ticket returns to `plan_pending`.
- **Delegation:** the delegate session may re-sign **T-IGX-036-P only**, under the same checks, additionally confirming that the plan's write set equals the amended WBS write set.
- **Not delegated** stays as listed above.
