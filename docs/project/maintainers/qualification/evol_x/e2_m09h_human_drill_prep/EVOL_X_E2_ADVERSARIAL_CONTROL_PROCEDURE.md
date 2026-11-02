# EVOL-X E2 — Adversarial control (M-09H)

**Procedure ID:** AC-M09H-01  
**Goal:** Evaluator cannot pass by copying `expected_cause_identities` from the P03 registry or from decoy “official” hashes.

## Controls

| ID | Control | Pass criterion |
|----|---------|----------------|
| AC-M09H-01a | **Registry blindness** | Evaluator attests (signed) they did not open `evol_x_p03_independent_replay/evol_m09_drill_case_registry.v1.json` or oracle key during C1–C3 |
| AC-M09H-01b | **Honeytoken resistance** | After all three sealed submissions, operator shows [AUDITOR_ONLY/adversarial_honeytokens.json](./AUDITOR_ONLY/adversarial_honeytokens.json); evaluator confirms **none** of those hashes were submitted |
| AC-M09H-01c | **Sealed timing integrity** | Worksheet UTC timestamps for attribution are monotonic per case and recorded before oracle review |

## Operator steps

1. Before C1: collect signed **AC-M09H-01a** attestation (worksheet section `adversarial_control.evaluator_attestation`).
2. After C3 submissions sealed: present honeytoken sheet; evaluator marks each token **NOT SUBMITTED** or flags if they used one (→ fail).
3. Record witness line: operator name, UTC, outcome.
4. Set `adversarial_control_passed` in final `evol_m09_human_drill.v2.json` to **true** only if AC-M09H-01a..c all pass.

## Fail closed

Any honeytoken match in submitted causes → `adversarial_control_passed: false` → Mode A final fails → do not force PASS.

## B6 auditor

Auditor verifies physical/digital seal timestamps, attestation signatures, and that diagnostic packets given to evaluator did not embed oracle values (spot-check C1–C3 cover sheets + tool JSON).
