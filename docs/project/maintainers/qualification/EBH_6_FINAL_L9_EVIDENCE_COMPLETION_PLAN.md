# EBH-6 — FINAL L9 EVIDENCE COMPLETION PLAN

| Field | Value |
| --- | --- |
| **Status** | **READY FOR INDEPENDENT PLAN AUDIT** |
| **Plan mode** | PLAN ONLY — no implementation, no Protocol 03 PASS, no EBH-6 CLOSED, no new roadmap stages |
| **Branch** | `development` |
| **STAGE_BASELINE / CURRENT HEAD** | `39a33d21a894f92fdc80ccc4f3f303006038d5a9` |
| **Operator anchor** | `EXPECTED_START_HEAD` = `39a33d21a894f92fdc80ccc4f3f303006038d5a9` |
| **Blocks** | **OD-EBH6-02** (plan acceptance); **L9 Protocol 03**; **OD-EBH6-08** (post-L9 only) |
| **Normative** | Roadmap §3.0.8, §5; ECAP §2.B B1–B6, L1–L9; VBN evidence spec §2.2 |

## Streszczenie wykonawcze (PL)

EBH-6 to finalna re-certyfikacja architektury na bieżącym HEAD — nie zastępuje EBH-5 ani CONFIG-X, lecz wymaga pełnego re-audytu hardcodingu vendor/backend w **16 warstwach** (§3.0.8), uczciwego rejestru `CORE_HARDCODING_FINDINGS_REGISTER` oraz **23 rodzin FRZ** z polami **independent disposition** (obecnie **16 × PENDING**). Mechaniczne bramki (`unclassified=0`, `prohibited unresolved=0`, CONFIG-X 54/54) są spełnione, ale wiersze **EBH6-LAYER-01..16** pozostają **OPEN / DISCOVERY REQUIRED** — plan zamyka je **prawdziwymi próbkami L9** (ścieżki, symbole, L3, klasy A–J), a nie sentinelami RESOLVED. SHA operacyjny to **`39a33d21…`**; **`badb7a086…`** jest wyłącznie **HISTORICAL**. EBH-5 parent **CLOSED** @ **OD-EBH5-08**; **51 NYQ** w macierzy replaceability oraz **DUP-X** (klasa D / duplikaty) pozostają poza zakresem zamknięcia EBH-6 — bez przenoszenia obowiązków mandatory na późniejsze etapy bez jawnego zapisu. Ten dokument definiuje batche **B-L9-1..6**, pytest L4 @ exact SHA, kryteria mechanical PASS dla audytora Protocol 03 oraz decyzje operatora (**OD-EBH6-01** tylko przy remediation produkcyjnej).

---

### L9-A Full-layer evidence (16 layers)

**Truth state (pre-cert):** [`EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md`](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md) — wszystkie `EBH6-LAYER-*` = **OPEN**; zamknięcie = **RESOLVED** lub jawne findingi PROHIBITED/ALLOWED z dowodem, nie zmiana statusu bez L9.

**Primary class per layer** (SSOT: `_findings_register._LAYER_DETECTION_CLASSES` — audytor weryfikuje **wszystkie** klasy A–J dotyczące warstwy, nie tylko primary):

| # | layer id | Primary class |
| --- | --- | --- |
| 01 | `contracts` | A |
| 02 | `semantic_core` | B |
| 03 | `application_services` | C |
| 04 | `composition_roots` | D |
| 05 | `governance` | E |
| 06 | `execution` | F |
| 07 | `integrations` | G |
| 08 | `state_persistence` | H |
| 09 | `reliability_recovery` | I |
| 10 | `observability_diagnostics` | J |
| 11 | `events_traceability` | A |
| 12 | `agent_distribution` | B |
| 13 | `rag_memory` | C |
| 14 | `runtime_hosting` | D |
| 15 | `tools_skills_capabilities` | E |
| 16 | `background_workers_infra_adapters` | F |

**L3 method (wspólny):** TRACE-X `PRODUCTION_COMPOSITION_SITE_REGISTRY` → `discover_sanctioned_production_composition_roots()`; EBH-5/6 `discover_composition_root_paths()`; `applications/**/*_wiring.py`; integracje `intergrax/integrations/providers/**/bundle.py` — narracja audytora: host/profile → wiring → resolver/factory → effect (bez live vendor/cloud).

| Layer | Paths / symbols to inspect | Composition / runtime entrypoints | Provider selection mechanisms | L3 production reachability | Evidence type (L1–L5) | A–J approach | Shadow / fallback detection | Exit criterion per layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **contracts** | `intergrax/contracts/**`, `intergrax/integrations/contracts/**`, public import surfaces in `tests/unit/contracts/` | Tier-0 contract packages; no runtime construction | N/A at contract layer — verify no vendor SDK types in public seams | From sanctioned app wiring to contract imports only (no direct provider import) | L1 import/AST gate; L2 contract files; L4 `test_tracing_public_contract.py` (sample) | **A:** vendor construction in contract modules; **F:** SDK leakage; **J:** extensibility claims vs RPL rows | Parallel “shadow contracts” (duplicate DTOs); implicit defaults in type aliases | `EBH6-LAYER-01` → **RESOLVED** or registered PROHIBITED with remediation; L9 attestation lists ≥3 production-effective samples or explicit `production_effective=no` with L3 trace |
| **semantic_core** | `intergrax/runtime/nexus/**`, decision/exposure seams (`AuthoritativeDecisionExposure`), core loop modules | Nexus orchestration; not Integrations materialization | Profile/catalog indirection only — no hardcoded backend enum in core | Nexus host composition → nexus_loop entry (sanctioned roots) | L1 pyright/architecture scans; L4 debt gate `EBH6-DEBT-Q2-D2` | **B:** hardcoded backend names; **C:** implicit defaults; **F:** typing pseudo-contracts | Silent fallback in core initialization | Layer sample doc + register rows; Q2-D2 remains RATIFIED unless OD-EBH6-01 opens remediation |
| **application_services** | `intergrax/applications/_shared/**` (memory_wiring, runtime_event_delivery_wiring) | Shared Tier-3 service wiring helpers | `IntegrationProfile` / resolver delegation — verify no local provider registry | App `*_wiring.py` → `_shared` helpers → contracts | L3 wiring trace; L1 grep for vendor literals | **A/C/I:** default construction in shared services | Duplicate resolver in `_shared` | No PROHIBITED-* unresolved on sampled paths; LAYER-02 RESOLVED |
| **composition_roots** | `COMPOSITION_ROOT_HINTS` + `discover_composition_root_paths()` — factory.py, profile registries, `existing_capability_configuration_facade.py` | Sanctioned composition owners per INTERGRAX principles §composition root | `resolve_from_profile`, Integration factory, LLM/RAG profile registries | Full L3 from TRACE-X sanctioned root through hint modules | L3 registry SSOT; L4 `test_ebh6_exit_sanctioned_composition_roots_non_empty` | **D:** shadow env/parallel registries; **H:** reachability of bypass imports | `EBH6-DEBT-BND-COMPAT-R3` lab path documented; no production shadow registry | LAYER-04 closed with D-class spot-check; DUP-X tracks duplicate selectors separately |
| **governance** | `intergrax/runtime/governance/**` | Governance plane — authorization minting, not execution | Config via governance settings contracts only | Production wiring → governance middleware attach | L4 GOV-X2 replay subset @ HEAD drift; L5 fail-closed samples from `test_npsc42_h1_governance_boundary_freeze.py` | **E:** shadow persistence of policy; **G:** vendor-locked diagnostics in governance path | Fail-open middleware (CTRL-X negative replay) | Independent disposition **GOV** moves PENDING → PASS/REPLAY with cited tests |
| **execution** | `intergrax/runtime/execution/**` | `ExecutionRuntime` canonical path (guard matrix #1–2) | Inference profile resolution — not vendor branch in core | Sanctioned execution host → `ExecutionRequest` → runtime | L4 `test_platform_execution_unification_u5_final_zero_bypass.py`; L1 `test_inference_profile_resolution.py` | **A/F/H:** vendor in execution core; bypass ingress | Second scheduler / alternate canonical path | LAYER-06 + family **EXE** evidence; guard matrix rows 1–2 cited @ SHA |
| **integrations** | `intergrax/integrations/**`, `providers/**/bundle.py` | Integration registry / factory; category enum closed-world size | `discover_integration_bundle_paths()`; CONFIG-X concern mapping | Wiring → factory → bundle → adapter boundary | L4 EBH-5 inventory replay; L1 EBH-5 RPL-G | **G/I:** operational coupling; inventory replay `EBH6-VBN-INVENTORY-REPLAY` | Shadow provider registry (defer detailed closure **DUP-X**) | LAYER-07 sample proves adapter-boundary discipline; note DUP-X for class **D** duplicates |
| **state_persistence** | SQLite annex 38 paths @ HEAD (`EBH6-SQLITE-*`); persistence providers under `intergrax/integrations/providers/` | Persistence contract → provider (guard matrix #6) | Sanctioned store selection via Integration profile | L3 to production-effective sqlite touchpoints + contract stores | L4 `test_ebh6_exit_sqlite_annex_accountability`; L8 N/A unless RPL row qualified | **E/H:** shadow stores; **C:** silent sqlite default | Unauthorized local JSON/SQLite outside annex | LAYER-08 RESOLVED via annex + ≥1 non-sqlite persistence sample trace |
| **reliability_recovery** | `intergrax/runtime/resilience/**`, recovery plane modules | Single recovery owner (guard #5) | Retry/recovery policies via platform contracts | Execution host → recovery plane wiring | L4 recovery qualification gates (NPSC5E family) | **I:** install-time default recovery store | Alternate recovery executor in plugin | LAYER-09 + **REC** family replay @ HEAD |
| **observability_diagnostics** | `intergrax/runtime/observability/**`, OTLP export paths | Non-authoritative export plane | Exporter selection via integration profile / qualified rows | Production OTLP concern from EBH-5 matrix replay | L4 `test_ebh5_exit_qualified_rows_have_behavioral_refs`; L1 class **G** | **G:** vendor-locked health/metrics without ratification | Silent fallback exporter | LAYER-10 class-**G** production sample required (pre-cert gap) |
| **events_traceability** | Event delivery wiring; trace correlation modules; `tests/qualification/trace_x/` | `runtime_event_delivery_wiring.py`; TRACE-X registry | Broker/backend via contracts only | Sanctioned root → event delivery wiring | L3 TRACE-X; L4 trace_x qualification subset | **A/D:** hardcoded broker client in core path | Duplicate provenance store | LAYER-11 RESOLVED with correlation spine narrative |
| **agent_distribution** | `agents/**` registry/assembly; distribution contracts | Agent OS harness boundaries (Tier-2) | Registry resolution — no apps importing private agent internals | Hosted agent profile → registry | L1 tier import gate; L4 distribution tests if in catalog | **B:** shadow agent registry | Bypass of sanctioned assembly | LAYER-12 sample + **HRN** partial overlap |
| **rag_memory** | `intergrax/rag/**`, `intergrax/memory/**`, embedding/rerank profile registries | RAG composition via Integration (EBH-2G evidence replay) | `intergrax/rag/embedding/registry/profile.py` — no RAG-local backend registry | App wiring → RAG services → Integration materialization | L4 EBH-2G mechanical gates replay; L1 RAG boundary gate | **C/J:** vector backend implicit default; extensibility vs 51 NYQ | Silent in-memory vector fallback | LAYER-13 documents **J** limitation (51 NYQ → EBH-7); no false global RPL PASS |
| **runtime_hosting** | Runtime host modules, hosting substrate adapters | Host composition profiles | Hosting provider via configuration contracts | Production host profile → runtime bootstrap | L3 hosting wiring from sanctioned sites | **D/I:** env shortcut host selection | Shadow host profile registry | LAYER-14 RESOLVED |
| **tools_skills_capabilities** | `intergrax/tools/**`, skills registry, `tools/registry/wiring.py` | Tool SPI + wiring | Tool provider resolution via registry | Wiring → tool registry → SPI | L4 plugin architecture gates (guard #11) | **J:** tool backend swap requires core change | Duplicate tool dispatchers | LAYER-15 + PLG replay cross-check |
| **background_workers_infra_adapters** | Worker entrypoints, infra adapter packages, background identity | Worker composition; no plugin-owned lifecycle | Queue/backend via integration contracts | Worker host wiring (if production-effective) | L1 worker module census; L4 identity authority gates | **H:** background bypass of governance | Silent queue fallback | LAYER-16 + ingress census (pre-cert **H** gap) |

**Deliverables L9-A:** zaktualizowany [`EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md`](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) + `_findings_register.py` (zamknięte LAYER rows); sekcja w [`EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md`](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) z tabelą próbek; auditor L9 narrative w przyszłym Protocol 03 record (poza tym planem).

---

### L9-B Architecture-family recertification (23 families)

**SSOT:** [`tests/qualification/ebh_6/_family_catalog.py`](../../../tests/qualification/ebh_6/_family_catalog.py) · matrix: [`EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md`](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md).

**Prioritized:** **16 × PENDING** (wszystkie poza REPLAY PASS / N/A).

| family_id | Current disposition | Existing proof (tests + qual doc @ SHA) | Gap | Minimal additional evidence (reuse first) | Acceptance rule | Independent disposition field (target) |
| --- | --- | --- | --- | --- | --- | --- |
| **BND** | REPLAY PASS | `test_hardening_3_layer_boundary_gate.py::test_tier0_must_not_import_agents_or_applications` @ `39a33d21…`; matrix @ HEAD | Current-HEAD drift vs EBH-2* scoped closures | Re-run tier gate + note DEBT-BND-COMPAT-R3 lab-only | No new PROVEN boundary violation on production path | **REPLAY PASS** (maintain) |
| **OWN** | PENDING | Catalog `test_ebh6_family_catalog_covers_applicable_themes` only | **EBH6-BLK-L9-OWN-01** — no ownership rescan | Reuse INTERGRAX principles ownership matrix + spot-check `intergrax/runtime/`, `intergrax/contracts/` for duplicate semantic owners (L1/L2 narrative) | No new duplicate semantic owner @ HEAD | **REPLAY PASS** or **PENDING→PASS** with L9 sign-off |
| **CTR** | PENDING | `test_ebh6_no_prohibited_unresolved_in_scope` | Register integrity ≠ full CTR rescan | Reuse `test_hardening_*` contract gates + public contract import boundaries @ HEAD | No PROHIBITED contract leakage unresolved | **PASS** with cited gates |
| **TYP** | PENDING | `test_ebh6_tracked_debt_findings_ratified` (Q2-D2) | Public seam typing not fully rescanned | Reuse GOV-X2 typing evidence + Q2-D2 RATIFIED row | Q2-D2 RATIFIED or remediated under OD-EBH6-01 | **N/A WITH EVIDENCE** for debt; else **PASS** |
| **PLG** | REPLAY PASS | `test_ebh5_exit_inventory_complete` | Regression vs EBH-5 only | Re-run EBH-5 mech gate @ HEAD | No global PASS promotion | **REPLAY PASS** |
| **RPL** | REPLAY PASS | `test_ebh5_rpl_g_config_x_active_blockers_remain_empty` | 51 NYQ documented | EBH-5 matrix replay doc SHA `33ecf19…` | Hardcoding re-audit only; matrix drift documented | **REPLAY PASS** |
| **EXE** | PENDING | `test_ebh6_family_row_count_closed_world` | **EBH6-BLK-L9-EXE-01** | Guard matrix #1–2 pytest bundle @ HEAD | No shadow execution path regression | **REPLAY PASS** with gate list |
| **GOV** | PENDING | family_row_count only | **EBH6-BLK-L9-GOV-01** | Subset GOV-X2 qualification replay (offline) | No new bypass | **REPLAY PASS** |
| **STA** | PENDING | `test_ebh6_exit_sqlite_annex_accountability` | SQLite slice only | Annex + persistence guard #6 replay | SQLite classified; no new shadow store | **PASS** (scoped STA) |
| **REC** | PENDING | **EBH6-BLK-L9-REC-01** | No recovery replay @ HEAD | Guard matrix #4–5 tests @ HEAD | Drift-only replay | **REPLAY PASS** |
| **CMP** | N/A — WITH EVIDENCE | COMPAT-X CLOSED historical | No EBH-6 consumer change | Document drift-only | Policy unchanged | **N/A — WITH EVIDENCE** (maintain) |
| **CFG** | REPLAY PASS | `test_ebh6_config_x_all_concerns_mapped_at_head` (54/54); CONFIG-X @ `fac9a43f…` | HEAD reconciliation only | Existing CFG test + neutrality audit | 54/54 mapped; historical PASS preserved | **REPLAY PASS** |
| **SEC** | PENDING | CTRL-X spot @ `318a4d83…` | **EBH6-BLK-L9-SEC-01** current-HEAD gap | `test_ctrl_x_r1_security_composition.py` fail-closed sample @ HEAD if unchanged | Spot-check — no fail-open regression | **REPLAY PASS** or scoped **PASS** |
| **REL** | PENDING | **EBH6-BLK-L9-REL-01** | No REL replay | EE-B1.1 / capacity gates (guard #10, #16) | Selected guard replay | **REPLAY PASS** |
| **CTL** | REPLAY PASS | `test_r2_mw_05_fail_open_rejected_before_attach` @ CTRL-X SHA | Historical SHA drift note | Re-run @ HEAD if middleware unchanged | Drift check only | **REPLAY PASS** |
| **OBS** | PENDING | `test_ebh5_exit_qualified_rows_have_behavioral_refs` partial | Class-G L9 sample | OTLP qualified row + observability export gate | Non-authoritative export preserved | **REPLAY PASS** + L9 OBS narrative |
| **TRC** | PENDING | `test_ebh6_exit_sanctioned_composition_roots_non_empty` | Wiring-only | TRACE-X qual subset + sanctioned roots table | Sanctioned roots wired for L3 | **PASS** with TRACE-X cite |
| **PRD** | N/A — WITH EVIDENCE | PROD-Q live OUT OF SCOPE | Live cloud | Roadmap Path B explicit exclusion | Live slice excluded from EBH-6 closure | **N/A — WITH EVIDENCE** |
| **REG** | PENDING | `test_ebh6_exit_mandatory_docs_present` | QUAL-X not closed | EBH-6 gates only | EBH-6 gates only — not QUAL-X closure | **PENDING→REPLAY PASS** for EBH-6 scope |
| **HRN** | PENDING | **EBH6-BLK-L9-HRN-01** | Harness subset | HARNESS-W5/W6 scoped tests replay | Subset replay | **REPLAY PASS** |
| **TEN** | PENDING | **EBH6-BLK-L9-TEN-01** | No global TEN promotion | `test_tenant_scope.py` spot-check | No global TEN promotion | **REPLAY PASS** scoped |
| **DOC** | PENDING | Mandatory doc presence | Content freshness = L9 | This plan + ledger @ `39a33d21…`; auditor reads dossier | Dossier + matrix docs present and SHA-consistent | **PASS** after plan audit + doc review |
| **VBN** | PENDING | `test_ebh6_findings_unclassified_zero`, `test_ebh6_prohibited_unresolved_zero` | Mechanical ≠ A–J completion | Full L9-A layer closure + neutrality rollup | unclassified=0; prohibited unresolved=0; L9 A–J attested | **PASS** only when L9-A complete |

**Qual doc SHA for matrix:** mechanical mirror @ **`39a33d21…`** after plan acceptance publication (**PUBLICATION_SHA** TBD at commit); supersede stale `716efd67…` references in ledger during **EBH6-L9-003**.

---

### L9-C Scope reconciliation

| Obligation | EBH-6 | DUP-X | EBH-7 | PROD-Q | Roadmap citation |
| --- | --- | --- | --- | --- | --- |
| Full-layer hardcoding re-audit (16 layers) | **PRIMARY** — must close LAYER samples | Supports class **D** duplicate/shadow selectors — **not** substitute for layer audit | Convergence — zero unresolved PROHIBITED in frozen scope | Not owner | §3.0.8 layers + EBH-6 mechanical exit |
| `CORE_HARDCODING_FINDINGS_REGISTER` | **PRIMARY** | Adds duplicate-finding rows when stage runs | Re-validates prohibited=0 platform-wide | N/A | VBN spec §2.2 |
| CONFIG-X reconciliation | **PRIMARY** (54/54 @ HEAD) | May surface new shadow config concerns | ROADMAP-REPLAY-X replays PASS vs HEAD | N/A | §3.0.8 CONFIG-X preservation |
| Architecture-family recertification (23) | **PRIMARY** — independent disposition | Overlap FRZ families per §2.0.2 table | Whole-platform FRZ convergence | PRD family N/A live | Roadmap §2.0.2 EBH-6 row |
| 51 RPL **NOT YET QUALIFIED** | Document limitation; **no deferral of mandatory layer findings** | N/A | **PRIMARY** for RPL-H closure | N/A | §3.0.8 stage table EBH-7 |
| Duplicate mechanisms / bypass | Spot-check in L9-A; **mandatory closure DUP-X** | **PRIMARY** | EBH-7 assumes DUP-X CLOSED | N/A | §3.0.8 + DUP-X stage § |
| Live provider / cloud qualification | **OUT OF SCOPE** (FRZ-PRD live OPEN) | N/A | N/A | **PRIMARY** Path B | §1.2, §2.0.3, checklist Path B |
| Global **FRZ PASS** | **Forbidden** — family disposition only | Forbidden | Target stage for many FRZ rows | Live FRZ-PRD/SEC portions | Checklist intro |
| Protocol 03 L9 audit | **PRIMARY** parent closure gate | Not started | Future platform audit | N/A | ECAP §2.B |

**Rule:** Mandatory EBH-6 findings (16 layers, VBN rollup, CONFIG-X mapped) **cannot** be marked “deferred to EBH-7/DUP-X” without a registered **OPEN** finding and roadmap-aligned stage owner — **DUP-X** owns duplicates; **EBH-7** owns 51 NYQ and platform-wide prohibited closure; **EBH-6** must still produce honest LAYER evidence.

---

### L9-D Evidence freshness

| Label | SHA | Role | Freshness |
| --- | --- | --- | --- |
| **STAGE_BASELINE** | `39a33d21a894f92fdc80ccc4f3f303006038d5a9` | Operator `EXPECTED_START_HEAD`; git containment anchor | **CURRENT** |
| **IMPLEMENTATION_SHA** | `39a33d21a894f92fdc80ccc4f3f303006038d5a9` | Code audited for L9 | **CURRENT** |
| **CERTIFICATION_HANDOFF_SHA** | `39a33d21a894f92fdc80ccc4f3f303006038d5a9` | Handoff pack baseline (update [`EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md`](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md)) | **CURRENT** |
| **TEST_EXECUTION_SHA** | `39a33d21a894f92fdc80ccc4f3f303006038d5a9` | Pytest L4 bundle @ run time | **CURRENT** (set at execution) |
| **PUBLICATION_SHA** | *(set at operator-authorized commit of plan + ledger)* | Doc/qual tree publication | **TBD** |
| **PRE-CERT PUBLICATION_SHA** | `a7d133a91cebc4c7cd8c5f265587168f05df5ee4` | Prior pre-cert doc batch | **HISTORICAL** |
| **Pre-cert IMPLEMENTATION** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` | Superseded by `39a33d21…` | **HISTORICAL** |
| **P0–P6 publication tree** | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` | First qual artifact commit | **HISTORICAL** |
| **badb7a086…** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` | Local batch — **must not cite as HEAD** | **HISTORICAL — superseded** |
| **EBH-5 independent audit** | `33ecf19cc020b112d9e586c87b49e819202f5d6d` | OD-EBH5-07 scoped PASS | **HISTORICAL — scoped PASS** |
| **EBH-5 parent closure** | OD-EBH5-08 APPROVED 2026-10-28 | Parent **CLOSED — SCOPED** | **Accepted** |
| **CONFIG-X PASS** | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` | FRZ-CFG-01..08 historical | **HISTORICAL — preserved** |
| **CTRL-X audit** | `318a4d835534ef5c48575f4649551c957dacd8ea` | Control-plane CLOSED | **HISTORICAL** |

**Actions:** [`EBH_6_EVIDENCE_FRESHNESS_LEDGER.md`](EBH_6_EVIDENCE_FRESHNESS_LEDGER.md) — align all CURRENT rows to `39a33d21…`; demote `badb7a086…` explicitly; update `_evidence_manifest.EXPECTED_START_HEAD` in implementation wave **only** after **OD-EBH6-01** if code change required (plan recommends doc-only SHA alignment first).

**Containment checks @ handoff:**

```powershell
git checkout development
git rev-parse HEAD  # expect 39a33d21…
git merge-base --is-ancestor 33ecf19cc020b112d9e586c87b49e819202f5d6d HEAD
git merge-base --is-ancestor 39a33d21a894f92fdc80ccc4f3f303006038d5a9 HEAD
```

---

### L9-E Qualification

**Constraints:** no heavy/cloud/live vendor; Windows dev; single pytest invocation per verification step ([`intergrax-test-process-budget.mdc`](../../../.cursor/rules/intergrax-test-process-budget.mdc)).

#### Consolidated pytest (L4 @ TEST_EXECUTION_SHA)

```powershell
New-Item -ItemType Directory -Force -Path .tmp/session/ebh-6-l9
uv run pytest `
  tests/qualification/ebh_6/ `
  tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py `
  tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py `
  tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py `
  -q 2>&1 | Tee-Object -FilePath .tmp/session/ebh-6-l9/pytest.log
```

| Batch | Gate bundle | Maps to |
| --- | --- | --- |
| **B-L9-4a** | `tests/qualification/ebh_6/` (4 modules) | Mechanical §3.0.8 exit + register integrity + CONFIG-X + family catalog |
| **B-L9-4b** | EBH-5 mech + RPL-G (above) | RPL/PLG replay @ HEAD |
| **B-L9-4c** | Tier boundary gate (optional extend with guard-matrix picks per family) | BND + post-freeze guard matrix #12 provider neutrality |

**Static gate bundles (L1):** post-freeze [`INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md`](INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md) — reuse representative gates per family (§ Guard Family Matrix rows 1–18); **no new production code** in plan mode.

**When OD-EBH6-01 required:** any **PROHIBITED-*** finding on production-effective path needing code remediation; register schema change that alters production behavior; fixing `_evidence_manifest.EXPECTED_START_HEAD` mismatch if tied to product fix; **not** required for doc-only L9 narratives, ledger SHA updates, or pytest-only replays @ HEAD.

---

### L9-F Protocol 03 mechanical PASS criteria

Independent auditor checklist (ECAP §2.B **B1–B6** + L9 attestation) — **agent must not self-PASS**.

| # | Criterion | Verification |
| --- | --- | --- |
| B1 | Closed-world inventory | [`EBH_6_ENTERPRISE_SURFACE_INVENTORY.md`](EBH_6_ENTERPRISE_SURFACE_INVENTORY.md) + `_closed_world_discovery.py` non-empty surfaces |
| B2 | Applicable FRZ-* | 23 families in catalog; checklist rows OPEN — **no global PASS rewrite** |
| B3 | Positive proofs | L4 pytest log @ **exact TEST_EXECUTION_SHA**; family gate_test_ids executed or waived with evidence |
| B4 | Adversarial negatives | Fail-closed samples cited (governance, CTRL-X middleware, persistence failure contracts) |
| B5 | Execution bundle | `.tmp/session/ebh-6-l9/pytest.log` + SHA ladder in freshness ledger |
| B6 | Independent acceptance | **`EBH_6_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`** created by auditor/operator only; roadmap §5 ledger updated by operator |
| R1 | `unclassified` = 0 | `test_ebh6_findings_unclassified_zero` |
| R2 | `prohibited unresolved` = 0 in frozen scope | `test_ebh6_prohibited_unresolved_zero` |
| R3 | 16 layers evidenced | All `EBH6-LAYER-*` **not** DISCOVERY_REQUIRED / OPEN |
| R4 | 23 families dispositioned | Each row **REPLAY PASS**, **PASS**, or **N/A — WITH EVIDENCE** — zero **PENDING** |
| R5 | CONFIG-X reconciled | 54/54 @ HEAD; historical `fac9a43f…` preserved |
| R6 | Negatives documented | Pre-cert limitations (51 NYQ, DUP-X, PROD-Q live) still true |
| R7 | Exact SHA | All claims cite `39a33d21…` (or newer operator-published SHA) — **not** `badb7a086…` as current |
| R8 | No agent self-PASS | No OD-EBH6-08; no EBH-6 CLOSED banner by implementer |

**SYS-INV / principles (L2):** cross-layer rules indexed in [`SYSTEM_INVARIANTS.md`](../../technical/guides/SYSTEM_INVARIANTS.md) (`SYS-INV-*`) — map per ECAP §3 family subsections (**BND** → tier boundaries; **OWN** → exactly-one owner; **CFG** → configured vs effective; **VBN** → provider boundary / no shadow selection). Auditor greps cited IDs at SHA; full file load optional.

---

### Work breakdown

| Task ID | Batch | Description | Depends on |
| --- | --- | --- | --- |
| **EBH6-L9-001** | B-L9-1 | Operator review this plan → **OD-EBH6-02** decision | — |
| **EBH6-L9-002** | B-L9-1 | Publish plan + refresh freshness ledger CURRENT = `39a33d21…`; remove `badb7a086…` as current | EBH6-L9-001 |
| **EBH6-L9-003** | B-L9-1 | Align handoff/dossier/pre-cert doc SHA banners to ladder | EBH6-L9-002 |
| **EBH6-L9-004** | B-L9-2 | Layer 01–04 sampling (contracts → composition_roots) — update register | EBH6-L9-002 |
| **EBH6-L9-005** | B-L9-2 | Layer 05–08 sampling (governance → state_persistence) | EBH6-L9-004 |
| **EBH6-L9-006** | B-L9-2 | Layer 09–12 sampling (reliability → agent_distribution) | EBH6-L9-005 |
| **EBH6-L9-007** | B-L9-2 | Layer 13–16 sampling (rag_memory → background_workers) | EBH6-L9-006 |
| **EBH6-L9-008** | B-L9-3 | Disposition **OWN, CTR, TYP, EXE, GOV** (PENDING priority) | EBH6-L9-004..007 |
| **EBH6-L9-009** | B-L9-3 | Disposition **STA, REC, SEC, REL, OBS, TRC** | EBH6-L9-008 |
| **EBH6-L9-010** | B-L9-3 | Disposition **REG, HRN, TEN, DOC, VBN** | EBH6-L9-009 |
| **EBH6-L9-011** | B-L9-4 | Run consolidated pytest @ HEAD → log `.tmp/session/ebh-6-l9/` | EBH6-L9-010 |
| **EBH6-L9-012** | B-L9-4 | Update TEST_EXECUTION_SHA + matrix/disposition fields in catalog SSOT | EBH6-L9-011 |
| **EBH6-L9-013** | B-L9-5 | Independent L9 audit (Protocol 03) — external auditor | EBH6-L9-011, EBH6-L9-007 |
| **EBH6-L9-014** | B-L9-5 | Create Protocol 03 PASS record or REJECT with findings | EBH6-L9-013 |
| **EBH6-L9-015** | B-L9-6 | **OD-EBH6-08** / parent CLOSED — operator only, post PASS | EBH6-L9-014 |

**Batch summary:**

| Batch | Focus |
| --- | --- |
| **B-L9-1** | Plan audit, SHA ladder, doc alignment |
| **B-L9-2** | 16-layer real evidence (close EBH6-LAYER-*) |
| **B-L9-3** | 16 PENDING FRZ families → disposition |
| **B-L9-4** | Pytest L4 + SSOT updates |
| **B-L9-5** | Independent Protocol 03 |
| **B-L9-6** | Formal closure (forbidden in plan/implement mode) |

---

### Operator decisions

| Decision | Qualification-only (no OD-EBH6-01) | Requires OD-EBH6-01 (`OPERATOR APPROVED — EBH-6 IMPLEMENTATION WAVE OPEN`) |
| --- | --- | --- |
| Accept this plan (**OD-EBH6-02**) | **Yes** | No |
| SHA ledger / doc banner updates | **Yes** | No |
| L9 layer sampling narratives + register row updates (no code) | **Yes** | No |
| Pytest replay @ HEAD | **Yes** | No |
| Family disposition doc + catalog `independent_disposition` enum updates | **Yes** | No |
| Remediate new **PROHIBITED-*** on production path | No | **Yes** |
| Product code for shadow registry / fallback removal | No | **Yes** |
| `_evidence_manifest.EXPECTED_START_HEAD` code constant change | Prefer doc-first | **Yes** if bundled with code wave |
| **OD-EBH6-08** / EBH-6 CLOSED | No — post L9 PASS only | N/A |

**Canonical wave phrase (not recorded):** `OPERATOR APPROVED — EBH-6 IMPLEMENTATION WAVE OPEN` — see [`EBH_6_OPERATOR_DECISION_REGISTER.md`](EBH_6_OPERATOR_DECISION_REGISTER.md).

---

## References

- [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) — EBH-6, §3.0.8, §5
- [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)
- [`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) — §2.B, L1–L9
- [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
- [`VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md`](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md)
- [`INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md`](INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md)
- EBH-6 qualification pack: `EBH_6_*.md`, `tests/qualification/ebh_6/`

---

**Plan author:** Cursor agent (plan-only). **Not** an independent audit PASS. **Next gate:** **OD-EBH6-02** independent plan audit.
