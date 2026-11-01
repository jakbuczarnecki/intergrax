# EVOL-X W1 — Independent review worksheet (AF-1..AF-8)

**Audit snapshot SHA (pre–F-batch):** `54e0de377644fe19156a2f8497efa6cd10d7fdd2`  
**Implementer attestation:** not independent acceptance (ECAP E9).

| Gate | Required evidence | Path / section | Finding | Blocker? | Operator signature |
|------|-------------------|----------------|---------|----------|-------------------|
| **AF-1** | Normative 10/10/8/11 | [EVOL_X_NORMATIVE_REQUIREMENTS_MATRIX.md](EVOL_X_NORMATIVE_REQUIREMENTS_MATRIX.md) | PASS | No | — |
| **AF-2** | Ownership ADR | [EVOL_X_CHANGE_SAFETY.md](../../../capabilities/architecture/EVOL_X_CHANGE_SAFETY.md) | Draft PASS; Option C proposed | Non-blocking | OD-EVOL-01 |
| **AF-3** | Zones + 10-Q | Same | PASS | No | — |
| **AF-4** | N register + N-18 table | [EVOL_X_N_VERIFICATION_REGISTER_W1.md](EVOL_X_N_VERIFICATION_REGISTER_W1.md), [EVOL_X_N18_…](EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md) | 32 modules reconciled; INF-TREE baseline; OD-10 for CONSERVATIVE_FULL | Non-blocking W2 code | OD-EVOL-10 |
| **AF-5** | Contract freeze | [EVOL_X_CONTRACT_SPEC.md](EVOL_X_CONTRACT_SPEC.md) | Draft PASS | No | — |
| **AF-6** | Signed ODs | [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](EVOL_X_OPERATOR_DECISION_REGISTER_W1.md) | **PENDING** | **YES — W2** | All OD-EVOL* |
| **AF-7** | Baseline | [W1_DISCOVERY_BASELINE.md](W1_DISCOVERY_BASELINE.md) | PASS | No | — |
| **AF-8** | External architecture audit | Worksheet signed by **≠ implementer** | **PENDING** | **YES — W2** (or documented waiver) | Independent reviewer |

**AF-6 and AF-8 must remain PENDING until explicitly recorded.**

**W2 entry:** **NOT AUTHORIZED** by this worksheet.
