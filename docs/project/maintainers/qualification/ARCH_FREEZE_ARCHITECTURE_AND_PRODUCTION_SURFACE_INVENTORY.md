# ARCH-FREEZE architecture and production surface inventory

**Method (planning phase — partial):** ENT enterprise audit manifest v1 (85 checks) + 8 anchor surfaces @ `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a`.

**Coverage honesty:** 85+8 = manifest-driven ENT audit coverage, **not** complete production reachability. Full closed-world scan: **T-B1-PROD-REACH-SCAN** (see SSOT §3.1). ENT `FULL`/`FREEZE` executes 85 checks; scan extends inventory to ~7k production modules.

| SURFACE_ID | DOMAIN | SEMANTIC_OWNER | ENTRYPOINT | IMPLEMENTATION_PATH | CONTRACT | PRODUCTION_REACHABILITY | INVARIANTS | TESTS | EVIDENCE | GAPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SURF-COMP-ROOT | Composition | Integrations/Applications | IntegrationProfile.resolve_from_profile | intergrax/integrations/** | Integration contracts | PRODUCTION_EFFECTIVE | FRZ-BND/CTR | INT-CONFIG qual | scoped PASS | Formal OPEN promotion |
| SURF-NEXUS-LOOP | Execution | Execution/Nexus | NexusLoop | intergrax/runtime/nexus/nexus_loop.py | Host orchestration wiring | PRODUCTION_EFFECTIVE | FRZ-EXE | Nexus qual inv | HARNESS-* | — |
| SURF-ENT-AUDIT | Governance | ENT-AUDIT-X | run_enterprise_audit.py run | scripts/enterprise_audit/ | Audit manifest | PRODUCTION_EFFECTIVE | FRZ-AUD | 85 manifest checks | ENT TES | Formal OPEN |
| SURF-GOV-RUNTIME | Governance | Governance | GovernanceService | intergrax/runtime/governance/** | GOV contracts | PRODUCTION_EFFECTIVE | FRZ-GOV | GOV-X qual | PASS family GOV | — |
| SURF-TENANT | Tenancy | TENANT-X | tenant propagation chain | cross-tier | FRZ-TEN | PRODUCTION_EFFECTIVE | FRZ-TEN | TENANT_X FINAL | 12/12 PASS | FREEZE reval |
| SURF-EVIDENCE-PERSIST | Persistence | Events/Evidence | EvidencePersistencePort | intergrax/contracts/execution_evidence/** | Typed port | PRODUCTION_EFFECTIVE | FRZ-REC | STATE/compat gates | — | — |
| SURF-DEBUG-LAB | Execution lab | EE debug zone | build_debug_minimal_nexus_loop | debug_lab_nexus_loop.py | Lab wiring | LAB_ONLY | FRZ-BND-03 | layer gate | CHECKLIST debt stale @ HEAD | T-B2-BND03-RECONCILE |
| SURF-LKW-APP | Applications | applications/ | FastAPI routers | applications/** | App contracts | PRODUCTION_EFFECTIVE | FRZ-PRD | PROD-Q scoped | Path B split | — |


## Manifest-derived checks (85)
| check_id | category | entrypoint_kind | modes |
| --- | --- | --- | --- |
| ENT-A-BND-01 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-BND-02 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-BND-03 | architecture | python_script | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-OWN-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-OWN-02 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-OWN-03 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-CTR-01 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-CTR-02 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-CTR-03 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-TYP-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-TYP-02 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-TYP-03 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-DUP-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-DUP-02 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-DUP-03 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-PLG-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-PLG-02 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-PLG-03 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-VBN-01 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-VBN-02 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-VBN-03 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-GOV-01 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-GOV-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-GOV-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-EXE-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-EXE-02 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-EXE-03 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-OBS-01 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-OBS-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-OBS-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-STA-01 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-STA-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-STA-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-SEC-01 | architecture | python_script | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-SEC-02 | architecture | python_script | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-SEC-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REL-01 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REL-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REL-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REC-01 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REC-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REC-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-PRD-01 | architecture | python_script | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-PRD-02 | architecture | python_script | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-PRD-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-CMP-01 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-CMP-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-CMP-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-REG-01 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-REG-02 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-REG-03 | architecture | python_script | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-DOC-01 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-DOC-02 | architecture | python_script | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-DOC-03 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-TEN-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-TEN-02 | architecture | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-TEN-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-EVOL-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-EVOL-02 | architecture | pytest_file | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-EVOL-03 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-INV-01 | architecture | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| ENT-A-INV-02 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| ENT-A-INV-03 | architecture | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| TEN-IDENTITY | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-PROPAGATION | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-AUTHORITY | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-STATE | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-PROVIDER | tenant | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| TEN-CONFIG | tenant | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-CREDENTIALS | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-EVIDENCE | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-TRACE | tenant | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| TEN-RECOVERY | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-ASYNC | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| TEN-CROSS-TENANT | tenant | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| TEN-FAIL-CLOSED | tenant | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-INVENTORY-DRIFT | vbn | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-CORE-HARDCODING | vbn | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-SHADOW-SELECTION | vbn | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-SILENT-FALLBACK | vbn | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-CONTRACT-LEAKAGE | vbn | pytest_nodeid | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-SQLITE-EFFECTIVE | vbn | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-OBS-OPS-COUPLING | vbn | pytest_nodeid | CHANGED-SINCE,FAST,FREEZE,FULL |
| VBN-REPLACEABILITY-MATRIX | vbn | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
| VBN-REACHABILITY | vbn | pytest_file | FAST,FULL,FREEZE,CHANGED-SINCE |
