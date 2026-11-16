# PDC-X — FRZ coverage matrix (PDC-P2-10)

**Generated** by `scripts/qualification/pdc_x_requirements.py`. Existing FRZ identifiers only (OD-PDC-FRZ-01 Option A, approved 2026-11-12). Checklist cardinality verified: **225** criteria in **26** families. No status is changed here; promotion happens only at PDC-P9 through the checklist update protocol.

## Coverage by family

| Family | Criteria | Referenced by PDC | Not referenced |
| --- | ---: | ---: | --- |
| FRZ-AUD | 10 | 1 | FRZ-AUD-01, FRZ-AUD-02, FRZ-AUD-03, FRZ-AUD-04, FRZ-AUD-06, FRZ-AUD-07, FRZ-AUD-08, FRZ-AUD-09, FRZ-AUD-10 |
| FRZ-BND | 6 | 2 | FRZ-BND-01, FRZ-BND-02, FRZ-BND-03, FRZ-BND-06 |
| FRZ-CFG | 8 | 5 | FRZ-CFG-04, FRZ-CFG-06, FRZ-CFG-07 |
| FRZ-CMP | 8 | 5 | FRZ-CMP-04, FRZ-CMP-06, FRZ-CMP-07 |
| FRZ-CTL | 12 | 0 | FRZ-CTL-01, FRZ-CTL-02, FRZ-CTL-03, FRZ-CTL-04, FRZ-CTL-05, FRZ-CTL-06, FRZ-CTL-07, FRZ-CTL-08, FRZ-CTL-09, FRZ-CTL-10, FRZ-CTL-11, FRZ-CTL-12 |
| FRZ-CTR | 6 | 4 | FRZ-CTR-02, FRZ-CTR-03 |
| FRZ-DEBT | 6 | 3 | FRZ-DEBT-01, FRZ-DEBT-02, FRZ-DEBT-03 |
| FRZ-DOC | 6 | 4 | FRZ-DOC-04, FRZ-DOC-06 |
| FRZ-EVOL | 10 | 3 | FRZ-EVOL-01, FRZ-EVOL-02, FRZ-EVOL-03, FRZ-EVOL-05, FRZ-EVOL-08, FRZ-EVOL-09, FRZ-EVOL-10 |
| FRZ-EXE | 7 | 4 | FRZ-EXE-03, FRZ-EXE-04, FRZ-EXE-05 |
| FRZ-FRZ | 13 | 0 | FRZ-FRZ-01, FRZ-FRZ-02, FRZ-FRZ-03, FRZ-FRZ-04, FRZ-FRZ-05, FRZ-FRZ-06, FRZ-FRZ-07, FRZ-FRZ-08, FRZ-FRZ-09, FRZ-FRZ-10, FRZ-FRZ-11, FRZ-FRZ-12, FRZ-FRZ-13 |
| FRZ-GOV | 10 | 7 | FRZ-GOV-07, FRZ-GOV-09, FRZ-GOV-10 |
| FRZ-HRN | 8 | 0 | FRZ-HRN-01, FRZ-HRN-02, FRZ-HRN-03, FRZ-HRN-04, FRZ-HRN-05, FRZ-HRN-06, FRZ-HRN-07, FRZ-HRN-08 |
| FRZ-OBS | 7 | 3 | FRZ-OBS-02, FRZ-OBS-03, FRZ-OBS-04, FRZ-OBS-07 |
| FRZ-OWN | 5 | 5 | — |
| FRZ-PLG | 8 | 6 | FRZ-PLG-06, FRZ-PLG-07 |
| FRZ-PRD | 12 | 1 | FRZ-PRD-02, FRZ-PRD-03, FRZ-PRD-04, FRZ-PRD-05, FRZ-PRD-06, FRZ-PRD-07, FRZ-PRD-08, FRZ-PRD-09, FRZ-PRD-10, FRZ-PRD-11, FRZ-PRD-12 |
| FRZ-REC | 10 | 5 | FRZ-REC-05, FRZ-REC-06, FRZ-REC-07, FRZ-REC-08, FRZ-REC-09 |
| FRZ-REG | 10 | 5 | FRZ-REG-01, FRZ-REG-04, FRZ-REG-07, FRZ-REG-08, FRZ-REG-10 |
| FRZ-REL | 11 | 11 | — |
| FRZ-RPL | 4 | 3 | FRZ-RPL-03 |
| FRZ-SEC | 10 | 10 | — |
| FRZ-STA | 8 | 5 | FRZ-STA-02, FRZ-STA-04, FRZ-STA-08 |
| FRZ-TEN | 12 | 8 | FRZ-TEN-02, FRZ-TEN-03, FRZ-TEN-09, FRZ-TEN-10 |
| FRZ-TRC | 12 | 9 | FRZ-TRC-02, FRZ-TRC-07, FRZ-TRC-10 |
| FRZ-TYP | 6 | 2 | FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-04, FRZ-TYP-06 |

Criteria not referenced stay with their certifying stages or with ARCH-FREEZE itself; PDC-P3 records any exception.

## Criterion → PDC requirements

| FRZ | Criterion | Checklist status | PDC requirements |
| --- | --- | --- | --- |
| FRZ-AUD-05 | every report is attributable to exact repository SHA | PASS | PDC-D2-R018, PDC-D5-R013, PDC-D6-R020, PDC-D8-R002, PDC-D8-R009, PDC-D8-R016, PDC-D8-R019 |
| FRZ-BND-04 | private implementation leakage = 0 | OPEN | PDC-D4-R015 |
| FRZ-BND-05 | cross-layer construction only sanctioned composition | OPEN | PDC-D4-R002 |
| FRZ-CFG-01 | every configurable provider/vendor/model/backend selection is derived from an explicit typed platform configuration/profile contract, not a hard-coded production implementation choice | PASS | PDC-D3-R003 |
| FRZ-CFG-02 | missing or invalid required configuration cannot silently select a vendor, model, backend or provider; the mechanism fails closed or remains explicitly disabled | PASS | PDC-D2-R008, PDC-D3-R015, PDC-D6-R008, PDC-D6-R014, PDC-D6-R024 |
| FRZ-CFG-03 | configured state ≠ effective state is mechanically enforced: installed/available/configured capability does not become effective without the sanctioned activation/composition path | PASS | PDC-D1-R008 |
| FRZ-CFG-05 | provider/vendor/model/backend-specific endpoints, deployments, regions, credentials or secret references are supplied through approved configuration/secret boundaries and are not embedded as production semantic choices in core code | PASS | PDC-D6-R027 |
| FRZ-CFG-08 | regression gates mechanically prevent reintroduction of hard-coded provider/vendor/model/backend selection, silent fallback and activation without required configuration | PASS | PDC-D3-R019 |
| FRZ-CMP-01 | frozen contracts inventory exists | PASS | PDC-D4-R014 |
| FRZ-CMP-02 | versioning policy exists | PASS | PDC-D1-R001 |
| FRZ-CMP-03 | persisted schemas versioned | PASS | PDC-D5-R002, PDC-D8-R003 |
| FRZ-CMP-05 | plugin/provider compatibility policy exists | PASS | PDC-D1-R010, PDC-D1-R011 |
| FRZ-CMP-08 | compatibility shims not parallel authorities | PASS | PDC-D8-R017 |
| FRZ-CTR-01 | consumers depend on contracts | OPEN | PDC-D1-R001, PDC-D1-R002, PDC-D1-R006, PDC-D2-R004, PDC-D2-R007, PDC-D2-R010, PDC-D3-R001, PDC-D3-R006, PDC-D3-R022, PDC-D4-R001, PDC-D5-R001, PDC-D7-R001 |
| FRZ-CTR-04 | public/internal contracts classified | OPEN | PDC-D4-R014 |
| FRZ-CTR-05 | contract responsibility narrow/cohesive | OPEN | PDC-D1-R005 |
| FRZ-CTR-06 | contracts do not encode concrete vendor/runtime implementation | OPEN | PDC-D3-R001, PDC-D3-R019, PDC-D4-R015 |
| FRZ-DEBT-04 | PARTIAL inside frozen scope = 0 | OPEN | PDC-D2-R005 |
| FRZ-DEBT-05 | DEFERRED inside frozen scope = 0 | OPEN | PDC-D1-R018, PDC-D1-R019, PDC-D3-R020, PDC-D4-R011, PDC-D4-R012, PDC-D4-R013, PDC-D7-R006, PDC-D7-R007, PDC-D7-R008, PDC-D8-R010 |
| FRZ-DEBT-06 | every N/A/outside-scope item evidence-backed | OPEN | PDC-D1-R019, PDC-D4-R011 |
| FRZ-DOC-01 | canonical docs match code | OPEN | PDC-D1-R009, PDC-D4-R016, PDC-D5-R017, PDC-D7-R011, PDC-D7-R023 |
| FRZ-DOC-02 | no conflicting semantic authority | OPEN | PDC-D4-R019, PDC-D5-R016, PDC-D6-R021, PDC-D6-R025 |
| FRZ-DOC-03 | ownership docs current | OPEN | PDC-D1-R021, PDC-D7-R022, PDC-D7-R023, PDC-D8-R001 |
| FRZ-DOC-05 | public contracts manifest current | OPEN | PDC-D4-R014 |
| FRZ-EVOL-04 | contract/schema evolution at change time detects consumers (COMPAT policy applied) | PASS | PDC-D1-R010 |
| FRZ-EVOL-06 | regression test selection from impact graph with mandatory safety suites | PASS — SCOPED | PDC-D5-R010, PDC-D5-R011 |
| FRZ-EVOL-07 | deployment / rollback / roll-forward qualified or unsafe refusal explicit | PASS (EVOL-X SCOPED) | PDC-D1-R011, PDC-D1-R012 |
| FRZ-EXE-01 | exactly one legal platform execution authority | PASS | PDC-D4-R017 |
| FRZ-EXE-02 | all meaningful executable work enters canonical boundary | PASS | PDC-D4-R017, PDC-D7-R002 |
| FRZ-EXE-06 | execution identity canonical | PASS | PDC-D7-R016 |
| FRZ-EXE-07 | no alternate scheduler/worker execution authority | PASS | PDC-D7-R002, PDC-D7-R004, PDC-D7-R006 |
| FRZ-GOV-01 | Governance != Execution | PASS | PDC-D5-R014, PDC-D6-R003 |
| FRZ-GOV-02 | proposal != permission != execution | PASS | PDC-D5-R014 |
| FRZ-GOV-03 | child_authority subset parent_authority | PASS | PDC-D6-R004 |
| FRZ-GOV-04 | downstream scope narrowing only | PASS | PDC-D6-R004 |
| FRZ-GOV-05 | fresh authorization before meaningful effects | PASS | PDC-D1-R013, PDC-D6-R018, PDC-D7-R018, PDC-D8-R006, PDC-D6-R028 |
| FRZ-GOV-06 | missing HITL never approval | PASS | PDC-D5-R008 |
| FRZ-GOV-08 | policy/profile revision attributable | PASS | PDC-D3-R005 |
| FRZ-OBS-01 | canonical event/evidence spine | OPEN | PDC-D1-R017, PDC-D4-R009, PDC-D5-R013, PDC-D6-R020 |
| FRZ-OBS-05 | terminal state observable | OPEN | PDC-D1-R014, PDC-D4-R003, PDC-D4-R004, PDC-D4-R009, PDC-D7-R019 |
| FRZ-OBS-06 | side-effect paths observable | OPEN | PDC-D3-R010, PDC-D5-R012, PDC-D8-R016 |
| FRZ-OWN-01 | every major concern has exactly one semantic owner | OPEN | PDC-D1-R001, PDC-D2-R003, PDC-D2-R006, PDC-D7-R005, PDC-D7-R022, PDC-D8-R001, PDC-D8-R011 |
| FRZ-OWN-02 | one canonical contract per responsibility | OPEN | PDC-D2-R017, PDC-D4-R001, PDC-D8-R017, PDC-D8-R018 |
| FRZ-OWN-03 | one sanctioned composition owner | OPEN | PDC-D4-R002, PDC-D4-R020 |
| FRZ-OWN-04 | no shadow authority | OPEN | PDC-D1-R004, PDC-D6-R003 |
| FRZ-OWN-05 | no duplicated semantic mechanism | OPEN | PDC-D1-R004, PDC-D1-R020, PDC-D3-R023, PDC-D5-R005, PDC-D6-R013, PDC-D8-R017 |
| FRZ-PLG-01 | extensible mechanisms expose platform contracts | OPEN | PDC-D1-R005, PDC-D3-R002, PDC-D4-R013, PDC-D5-R005 |
| FRZ-PLG-02 | external structural implementation works without core patch | OPEN | PDC-D3-R020, PDC-D1-R022 |
| FRZ-PLG-03 | provider discovery has explicit owner | OPEN | PDC-D1-R020 |
| FRZ-PLG-04 | provider selection has explicit owner | OPEN | PDC-D3-R003 |
| FRZ-PLG-05 | activation/admission explicit | OPEN | PDC-D1-R003, PDC-D6-R016 |
| FRZ-PLG-08 | dynamic registration reversible/scoped when supported | OPEN | PDC-D1-R009, PDC-D1-R011, PDC-D1-R014, PDC-D1-R018 |
| FRZ-PRD-01 | production qualification independent of lab/harness | OPEN | PDC-D3-R021 |
| FRZ-REC-01 | crash recovery deterministic | PASS | PDC-D2-R013, PDC-D4-R007, PDC-D7-R003, PDC-D7-R015, PDC-D8-R007, PDC-D8-R014 |
| FRZ-REC-02 | restart recovery preserves authority | PASS | PDC-D1-R012, PDC-D1-R016, PDC-D4-R007, PDC-D4-R010 |
| FRZ-REC-03 | resume does not mint new identity incorrectly | PASS | PDC-D7-R015 |
| FRZ-REC-04 | replay cannot create divergent truth | PASS | PDC-D2-R011, PDC-D2-R012, PDC-D8-R013 |
| FRZ-REC-10 | partial/corrupt durable state is detected and cannot silently become accepted canonical truth | PASS | PDC-D2-R018, PDC-D8-R002 |
| FRZ-REG-02 | every corrected blocker has regression gate | OPEN | PDC-D1-R022, PDC-D4-R020, PDC-D6-R028 |
| FRZ-REG-03 | negative tests prove gates detect violations | OPEN | PDC-D1-R004, PDC-D2-R007 |
| FRZ-REG-05 | stale inventories = 0 | OPEN | PDC-D1-R021, PDC-D3-R024 |
| FRZ-REG-06 | deterministic qualification tests | OPEN | PDC-D5-R002, PDC-D5-R009 |
| FRZ-REG-09 | critical invariant protected by code/test, not docs only | OPEN | PDC-D3-R022, PDC-D5-R014, PDC-D6-R007 |
| FRZ-REL-01 | timeout ownership explicit | OPEN | PDC-D3-R012 |
| FRZ-REL-02 | retry ownership explicit | OPEN | PDC-D7-R009 |
| FRZ-REL-03 | idempotency ownership explicit | OPEN | PDC-D7-R010, PDC-D7-R011 |
| FRZ-REL-04 | cancellation propagates correctly | OPEN | PDC-D3-R011, PDC-D4-R005, PDC-D7-R014, PDC-D7-R020 |
| FRZ-REL-05 | external operation termination semantics explicit | OPEN | PDC-D3-R011 |
| FRZ-REL-06 | provider failure typed | OPEN | PDC-D3-R006, PDC-D3-R009, PDC-D3-R013, PDC-D4-R008 |
| FRZ-REL-07 | no hidden infinite retry/fallback | OPEN | PDC-D3-R004, PDC-D7-R009, PDC-D7-R013 |
| FRZ-REL-08 | bounded concurrency and work-admission semantics explicit | OPEN | PDC-D7-R021 |
| FRZ-REL-09 | queue/backpressure and overload behavior explicit | OPEN | PDC-D7-R012, PDC-D7-R013 |
| FRZ-REL-10 | resource saturation/exhaustion has deterministic reject/degrade/fail-closed behavior | OPEN | PDC-D2-R010, PDC-D3-R014, PDC-D4-R003, PDC-D8-R012 |
| FRZ-REL-11 | provider throttling/rate-limit handling cannot create unbounded retry/fallback or bypass | OPEN | PDC-D3-R004 |
| FRZ-RPL-01 | representative default provider replaceable | OPEN | PDC-D3-R016, PDC-D3-R024 |
| FRZ-RPL-02 | custom implementation proof exists | OPEN | PDC-D3-R017 |
| FRZ-RPL-04 | vendor/backend switch does not modify consumers | OPEN | PDC-D1-R015, PDC-D3-R016, PDC-D8-R015 |
| FRZ-SEC-01 | tenant isolation proven | OPEN | PDC-D2-R002, PDC-D6-R005, PDC-D6-R019 |
| FRZ-SEC-02 | secrets references / late resolution | OPEN | PDC-D6-R006, PDC-D6-R008, PDC-D6-R009, PDC-D6-R027 |
| FRZ-SEC-03 | raw secrets do not cross unrelated layers | OPEN | PDC-D6-R007 |
| FRZ-SEC-04 | production auth bypass = 0 | OPEN | PDC-D6-R002, PDC-D6-R017, PDC-D6-R026 |
| FRZ-SEC-05 | sandbox/isolation boundaries qualified | OPEN | PDC-D6-R010, PDC-D6-R011, PDC-D6-R012 |
| FRZ-SEC-06 | provider/plugin trust qualified | OPEN | PDC-D1-R003, PDC-D1-R013, PDC-D6-R016 |
| FRZ-SEC-07 | security claim bounded by explicit threat model/evidence | OPEN | PDC-D4-R019, PDC-D5-R006, PDC-D6-R021 |
| FRZ-SEC-08 | sensitive/tenant data propagation across platform/provider boundaries is explicit and qualified | OPEN | PDC-D2-R008, PDC-D2-R009, PDC-D2-R014, PDC-D6-R013, PDC-D6-R014, PDC-D6-R015, PDC-D8-R006, PDC-D8-R010 |
| FRZ-SEC-09 | retention / deletion / redaction responsibility is explicit for platform-owned durable or sensitive data | OPEN | PDC-D2-R015, PDC-D6-R022, PDC-D8-R008, PDC-D8-R009 |
| FRZ-SEC-10 | platform vs infrastructure responsibility for encryption/data residency/provider exposure is explicit and evidence-backed | OPEN | PDC-D6-R023 |
| FRZ-STA-01 | exactly one truth owner per state family | PASS | PDC-D2-R003, PDC-D4-R006 |
| FRZ-STA-03 | atomic/transaction boundaries explicit | PASS | PDC-D5-R018, PDC-D7-R012 |
| FRZ-STA-05 | stale state rejected | PASS | PDC-D2-R005 |
| FRZ-STA-06 | configured/effective/persisted state distinction | PASS | PDC-D1-R016, PDC-D2-R012 |
| FRZ-STA-07 | policy artifacts durable where required | PASS | PDC-D8-R007 |
| FRZ-TEN-01 | canonical tenant identity where tenant scope applies | PASS | PDC-D5-R015, PDC-D6-R001 |
| FRZ-TEN-04 | cross-tenant state/cache/persistence/memory/index access blocked without explicit governed cross-tenant semantics | PASS | PDC-D1-R007, PDC-D2-R002, PDC-D8-R005 |
| FRZ-TEN-05 | provider selection, configuration, profiles and effective bindings preserve tenant scope | PASS | PDC-D3-R018, PDC-D4-R018 |
| FRZ-TEN-06 | tenant-specific credentials/secrets references cannot resolve or reuse under another tenant context | PASS | PDC-D6-R006, PDC-D6-R009 |
| FRZ-TEN-07 | evidence, events, traces and diagnostics preserve tenant identity; cannot establish truth for another tenant | PASS | PDC-D5-R015 |
| FRZ-TEN-08 | checkpoint, retry, recovery, replay and restore preserve original tenant identity | PASS | PDC-D7-R017 |
| FRZ-TEN-11 | runtime extensions/providers/plugins cannot mint, rewrite or widen tenant scope beyond explicitly supplied authority | PASS | PDC-D1-R007, PDC-D6-R017 |
| FRZ-TEN-12 | executable adversarial tests prove tenant isolation across execution, state, provider/config, evidence and recovery critical paths | PASS | PDC-D6-R005, PDC-D6-R019 |
| FRZ-TRC-01 | execution causal chain reconstructable | PASS | PDC-D8-R004 |
| FRZ-TRC-03 | tool invocation attributable | PASS | PDC-D8-R004 |
| FRZ-TRC-04 | provider invocation attributable | PASS | PDC-D3-R005, PDC-D5-R004 |
| FRZ-TRC-05 | model call/context decision attributable | PASS | PDC-D2-R001, PDC-D2-R011, PDC-D2-R016, PDC-D5-R004 |
| FRZ-TRC-06 | side effect attributable to authorization | PASS | PDC-D6-R018 |
| FRZ-TRC-08 | profile revision attributable | PASS | PDC-D1-R017 |
| FRZ-TRC-09 | restart/resume preserves trace continuity | PASS | PDC-D2-R013 |
| FRZ-TRC-11 | configured/effective provenance reconstructable | PASS | PDC-D4-R010 |
| FRZ-TRC-12 | transport identity != runtime identity and mapping attributable | PASS | PDC-D7-R016 |
| FRZ-TYP-01 | semantic boundaries strongly typed | OPEN | PDC-D1-R002, PDC-D2-R004, PDC-D3-R007, PDC-D3-R008, PDC-D5-R001, PDC-D5-R003, PDC-D8-R018 |
| FRZ-TYP-05 | string-dispatch substitute for typed contract = 0 | OPEN | PDC-D3-R009, PDC-D5-R007 |
