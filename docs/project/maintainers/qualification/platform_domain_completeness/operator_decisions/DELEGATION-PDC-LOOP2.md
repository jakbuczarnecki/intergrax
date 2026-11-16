# DELEGATION-PDC-LOOP2 — Operator decision on the second repair loop after the P8 global audit

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-LOOP2 |
| **Stage** | PDC-X P8 (repair loop 2), before P8-04 |
| **Basis** | `DELEGATION-PDC-CLOSE.md` (decision 2: a second loop returns to the Operator); `DELEGATION-PDC-P7-REPAIR.md` (limits of delegate signing); `DELEGATION-PDC-P8-ACT.md`; `PDC_INDEPENDENT_AUDIT.md` section 13 (final addendum at `d39a520dd`) |
| **Date** | 2026-11-16 |

## Findings behind the question

- The final addendum of IA-GLOBAL rated the stage PARTIAL. Finding **G-23** (MEDIUM) is a confirmed product defect without an owner: the graph side resolves the active publication generation with a key that has no `publication_scope_id` (`intergrax/rag/graph/generation_visibility.py`, `intergrax/rag/graph/providers/inmemory_graph_store.py`), so after a re-ingest the graph evidence of the source is not visible.
- The EBH-7 L3 replay at pin `d39a520dd` (run 43738431866) failed: 2 failed, 720 passed, 9 skipped. The two failures are `test_f10_c4_041k_removing_the_entries_reopens_exactly_the_three_paths` and `test_f10_c4_041o_removing_the_entries_reopens_exactly_the_three_paths`. They append three paths to the canonical changed paths without removing duplicates, and at the new pin those paths are already in the range, so each reason occurs twice. This is a test defect, not a product defect. The run is **not** B6 evidence.

## Commission (transcription)

The orchestrator asked the Operator three questions (second loop, the 77 red incident scenario tests, the product MEDIUM findings E1–E11 and `trace_bridge.v1`). The Operator answered in English rendering: **"I approve the fixes, on condition that after they are introduced only the tests that covered the fixes are run, without re-running the long-running tests."** (original wording in Polish: "Zatwierdzam poprawki, ale warunkiem jest to, że po ich wprowadzeniu zostaną uruchomione tylko testy, które obejmowały poprawki, bez uruchamiania ponownie długich i długotrwałych testów.")

| Part | Decision |
| --- | --- |
| Second loop | Approved, one loop, no further loop. Two tickets: **T-IGX-041-P** (repair of G-23 in the two named files, with a negative test) and **T-IGX-041-Q** (deduplicate the appended paths in the K and O gate tests; no gate is weakened). |
| Condition | After the fixes only the tests that cover the fixes run: the tests added or changed by P and Q, and the reproduction tests named in G-23. The long suites are **not** re-run: no EBH-7 L3 replay, no new EBH-7 pin, no repeat of the full global audit pytest. The replay run 43738431866 therefore stays recorded as failed and "not B6 PASS"; the EBH-7 identity gate stays a known red class. |
| Delta audit | The audited SHA moves because P changes a product path (ACT-N11). A delta audit by a session that did not implement, sign or certify verifies P and Q with the targeted tests only and confirms that no other product path changed after `d39a520dd`. It does not repeat the broad pytest. |
| Not answered | The 77 red incident scenario tests and the product MEDIUM findings E1–E11 and `trace_bridge.v1` were **not** answered. They are not approved as exceptions and not repaired. They stay open and are carried unchanged into the P9 package, where the Operator decides them with the final signature. |

## Scope of the delegation

A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-041-P** and **T-IGX-041-Q** only, after confirming for each: the plan digest matches; the plan matches its batch row in `PDC_IMPLEMENTATION_WBS.md` (write sets equal, compared as lists); every factual sentence and file:line of the plan is true against the code; no gate is weakened or deleted; the write set of P contains only the two named files under `intergrax/rag/graph/` and its tests, and nothing under `intergrax/applications/_shared/`, `intergrax/contracts/`, `scripts/governance/` or `.github/`; the write set of Q contains only the two named test files; the plan adds no registry, scheduler, engine, trust, policy or audit authority; it states it is not a certification verdict; the plan states the targeted-test condition above; the implementer and the signer are different sessions.

Merge gate: authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md`: green CI, only the known red classes (IIAC activation staleness until the P8-04 rebind; the EBH-7 identity gate, which stays red under the condition above), head equals the verified SHA.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X closure line; lifting C10/C15; a change of the meaning of a gate; a third loop; any exception for the 77 scenario tests, E1–E11 or `trace_bridge.v1`; the verdict on B6 or certification.

## Separation

The implementer session and its subagents never write `Approved:` or activation lines. A delegate session does not implement or audit what it signs. The delta auditor never certifies and never fixes what it audits.
