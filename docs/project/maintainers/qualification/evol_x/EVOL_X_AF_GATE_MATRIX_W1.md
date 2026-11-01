# EVOL-X W1 — AF architecture-freeze gate matrix (reconciled)

| Gate | Required evidence | Document § | Outcome | Missing | Block? | Owner |
|------|-------------------|------------|---------|---------|--------|-------|
| **AF-1** | 10/10/8/11 norm rows | [EVOL_X_NORMATIVE_REQUIREMENTS_MATRIX.md](EVOL_X_NORMATIVE_REQUIREMENTS_MATRIX.md) | **PASS** | — | No | EVOL-X |
| **AF-2** | Ownership 0 conflicts | [EVOL_X_CHANGE_SAFETY.md](../../../capabilities/architecture/EVOL_X_CHANGE_SAFETY.md) | **PASS** (draft) | Operator ADR sign-off | Non-blocking | Operator |
| **AF-3** | Zone + 10-Q screen | Same | **PASS** | — | No | EVOL-X |
| **AF-4** | N-01..18 closed or blocker | [EVOL_X_N_VERIFICATION_REGISTER_W1.md](EVOL_X_N_VERIFICATION_REGISTER_W1.md), [EVOL_X_N18_…](EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md), [EVOL_X_N04_…](EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md) | **PASS with recorded blockers** | N-18 → OD-EVOL-10; N-02 partial | **Non-blocking for W2 code**; blocks naive CONSERVATIVE_FULL | Operator (OD-10) |
| **AF-5** | Contract spec | [EVOL_X_CONTRACT_SPEC.md](EVOL_X_CONTRACT_SPEC.md) | **PASS** (draft) | — | No | EVOL-X |
| **AF-6** | Signed ODs | [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](EVOL_X_OPERATOR_DECISION_REGISTER_W1.md) | **FAIL (pending)** | All signatures | **Yes — W2 entry** | Operator |
| **AF-7** | Baseline numbers | [W1_DISCOVERY_BASELINE.md](W1_DISCOVERY_BASELINE.md) | **PASS** | — | No | EVOL-X |
| **AF-8** | Independent architecture audit | External worksheet | **FAIL (pending)** | Reviewer + PASS record | **Yes — W2 entry** (or waiver) | Operator |

**AF-6 and AF-8 are not PASS** without signatures / independent audit.
