# DELEGATION-PDC-P7-REPIN — Operator decisions for the final EBH-7 re-pin after the P7 repairs

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P7-REPIN |
| **Stage** | PDC-X P7 closing, before P8 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md` (answers 4 and 8); `DELEGATION-PDC-WAVE2.md` (decisions 1, 3, 6, 7); `DELEGATION-PDC-P6-REPIN.md`; `DELEGATION-PDC-P7-REPAIR.md` |
| **Date** | 2026-11-15 |

## Commission (transcription)

The plan of T-IGX-041-J (final EBH-7 re-pin, batch B-P7-RP) found that the existing delegation records do not cover three points. The orchestrator asked the Operator one question with three parts; the Operator answered in English rendering: **"Yes, I approve (a)-(c)"** (original wording in Polish: "Tak, zatwierdzam (a)-(c)").

| Part | Decision |
| --- | --- |
| (a) | A delegate session may sign the plans of **T-IGX-041-J** (the re-pin itself) and of two preparatory tickets that must merge before the pin: **T-IGX-041-K** (add classifier entries to the B6-F10 canonical inputs for the three paths the classifier reports as UNKNOWN: `intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py`, `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py`; entries are added, the default classification never changes) and **T-IGX-041-L** (re-pin the COMPAT-X counters of debt item Z-D1 at the last product change, without weakening any gate). |
| (b) | The replay `unit-tests.yml` with `profile=ebh7_p3_l3` is dispatched at the new pin SHA `P` (the `development` tip taken after the last test or product merge). |
| (c) | Both SHA inputs of the replay (`ebh7_test_execution_sha`, `ebh7_evidence_publication_sha`) equal `P`, as for the previous pin `146e7acf1` (`DELEGATION-PDC-WAVE2.md` decision 1). |

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-041-J, T-IGX-041-K and T-IGX-041-L** only, after confirming for each: the plan digest matches; the plan matches its batch row in `PDC_IMPLEMENTATION_WBS.md` (write sets equal, compared as lists); every factual sentence of the plan is true against the code; no gate is weakened or deleted (classification entries and re-pins only, each with a negative test); the write set contains nothing under `intergrax/`, `agents/`, `applications/`, `scripts/governance/` and nothing under `.github/` except, for T-IGX-041-J only, the pin defaults of `.github/workflows/unit-tests.yml` named in its plan; the plan adds no registry, scheduler, engine, trust, policy or audit authority; it states it is not a certification verdict; the implementer and the signer are different sessions.
- **Ordering.** The tickets K and L and the signed records of J are merged **before** `P` is taken, because `governance/tickets/` and `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` are outside the executable-identity allowlist. After `P` only allowlisted paths change until the replay is recorded.
- **Merge gate.** The pull requests merge under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation staleness until the P8-04 rebind; the EBH-7 identity gate until this re-pin is recorded), head equals the verified SHA.
- The replay is dispatched once; the run id and the artifact SHA-256 values are recorded in a docs-only pull request that states "not B6 PASS".

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X closure line; lifting C10/C15; write-set amendments of these tickets; any change of an allowlist or of the meaning of a gate; the verdict on B6 or certification.

## Separation

The implementer session and its subagents never write `Approved:` or activation lines. A delegate session does not implement or audit what it signs. An auditor never certifies and never fixes what it audits.
