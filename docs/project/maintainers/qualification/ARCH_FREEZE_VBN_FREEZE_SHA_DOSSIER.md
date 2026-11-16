# ARCH-FREEZE — VBN freeze SHA dossier (§2.1–2.7 rollup, §3 SQLite annex)

**Status:** W2 DRAFT @ `332c5217c390202a911ad5592e88a8f366845587` (T-AF2-VBN-DOSSIER, ticket T-IGX-050-H) — **NOT** independently accepted · **NOT** FRZ-RPL promotion · **NOT** a freeze claim  
**Draft evidence SHA (tables below were checked at this commit):** `332c5217c390202a911ad5592e88a8f366845587` (`origin/development`)  
**Earlier R1 baseline (history, superseded as the reading SHA):** `4a1ab152dc86e97fb017278310c553064a664b23`  
**FREEZE_CERTIFICATION_SHA:** `TBD` (set at **T-B7-LOCK-FREEZE-SHA**; no acceptance claim is made at any SHA before it)  
**CLOSURE_DOC_HEAD (wave-1 SSOT):** `950097ea4bcfd5bba828f663bcd993b8bd1e1d63`  
**Authority:** [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](./VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2–§3 (schema) · WBS **T-B7-VBN-HEAD-DOSSIER** / **T-B7-VBN-INDEPENDENT-ACCEPT** · qual plan **L2-VBN-EBH5** / **L2-VBN-ENT**

**DAG note:** this file is the W2 draft of **T-AF2-VBN-DOSSIER** (plan MEP-AF2 v2.1, section H, step 1). The W5 final (ticket T-IGX-052-C, = **T-B7-VBN-HEAD-DOSSIER**) is regenerated at the pre-pin tip. **PT-Q13** supplies the certifying runs at M3. **PT-VBN = T-B7-VBN-INDEPENDENT-ACCEPT** (W7, after the lock) is the only step that can accept anything here: a new independent session gives per-artefact verdicts at evidence SHA = M3, independently ratifies each fixed substrate of §2.4, replays the 12 VBN checks with `ENT_AUDIT_X_SEMANTIC=1` and signs §2.7. **T-B7-PFP-ACTIVATE** precedes the lock.

## Draft scope and limits

- **What was done at the draft SHA.** Every path cited below was checked to exist in the tree. The counts in §2.1, §2.3, §2.4, §2.6 and §3 were read by importing the EBH-5 register modules and calling their pure accessor functions once, read-only, under the shared process lock (no pytest run, nothing written to the repository). They are **not** certifying runs: no RUN_ID exists at FREEZE_SHA.
- **What was not done.** No test suite was run. No behavioural proof was replayed. The reachability and sandbox-provider figures quoted from the closed-world result were measured at the older audit baseline `7960ae7f6c82da60ca9acdc0ceebf63810002a42` and are marked as such. The S1 figures are from a non-certifying diagnostic and are quoted with attribution.
- **No acceptance language.** "Present", "counted" and "recorded" below describe the draft SHA only. Nothing here closes **FIND-VBN-PARTIAL**, promotes FRZ-RPL-01..04 or admits ECAP §15.

---

## §2.1 — Vendor/backend dependency inventory

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_vendor_backend_inventory.py`](../../../../tests/qualification/ebh_5/_vendor_backend_inventory.py) |
| Closed-world validation | [`tests/qualification/ebh_5/_inventory_closed_world_validation.py`](../../../../tests/qualification/ebh_5/_inventory_closed_world_validation.py) |
| Mechanical tests (**L2-VBN-EBH5**) | `uv run pytest tests/qualification/ebh_5/ -q` |
| ENT drift gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_inventory_drift.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_inventory_drift.py) |
| Records @ draft SHA (read-only accessor call) | **69** concern records; `unclassified` = **0**; the closed-world validation function returned **0** errors; all 69 carry conclusion `ACCEPTED` (2 records are marked not production-effective) |
| Classification @ draft SHA | `REPLACEABLE — QUALIFIED` 48 · `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE` 17 · `INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED` 4 · `REPLACEABLE — NOT YET QUALIFIED` 0 |
| Roadmap domains 1–31 | 25 domains `COVERED`, 6 `N/A — WITH EVIDENCE` (4 cache, 10 distributed coordination, 19 logging, 20 metrics, 28 network/HTTP, 30 evaluation/analytics); the domain-by-domain annex is [ANNEX_D_VBN_DOMAINS_1_31.md](./ent_audit_x/ANNEX_D_VBN_DOMAINS_1_31.md) (it quotes 69 concerns, equal to the count above). Domain 30 is a bounded scan, not an absence claim (register text) |
| CONFIG-X historical 54 | The CONFIG-X inventory source holds 54 rows by static reading (19 literal records plus one template row expanded over 35 `IntegrationCategory` members). Equivalence with the 69 records is **not** assumed; the reconciliation module is [`_config_x_reconciliation.py`](../../../../tests/qualification/ebh_5/_config_x_reconciliation.py) |
| Status @ `332c5217c` | Registers present and internally consistent at the accessor level; **no** certifying RUN_ID @ FREEZE_SHA; `unclassified=0` acceptance **pending** independent replay @ FREEZE_SHA |
| Acceptance @ FREEZE_SHA | `unclassified=0`; closed-world concern count reconciled with CONFIG-X historical 54 without equivalence assumption; domains 1–31 accounted |

## §2.2 — Core hardcoding findings register

| Field | Value |
| --- | --- |
| EBH-5 integrity gates | [`test_ebh_5_mechanical_exit_gates.py`](../../../../tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py), [`test_ebh_5_proof_integrity_gates.py`](../../../../tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py), [`test_ebh_5_l2_integrity_gates.py`](../../../../tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py) |
| ENT gates | [`test_ent_a_vbn_core_hardcoding.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_core_hardcoding.py), [`test_ent_a_vbn_shadow_selection.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_shadow_selection.py), [`test_ent_a_vbn_silent_fallback.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_silent_fallback.py), [`test_ent_a_vbn_contract_leakage.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_contract_leakage.py) |
| Register documents | [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](./EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md); wave-1 engineering evidence [ARCH_FREEZE_B2_VBN_CURRENT_HEAD_EVIDENCE.md](./ARCH_FREEZE_B2_VBN_CURRENT_HEAD_EVIDENCE.md) @ `f7ce9dae` |
| Open inputs for this register (not classified here) | **A4-018** — direct `ollama` SDK use: the closed-world result lists 5 product import sites at the older baseline `7960ae7f6` (in the LLM adapter package, `intergrax/multimedia/`, `agents/model_routing_qualifier/` and one Tier-3 benchmark module); plan section E routes it to **T-AF2-RW-RUNTIME**. Whether any site is a sanctioned adapter boundary or a `PROHIBITED — CORE VENDOR HARDCODING` finding is for the independent classification at PT-VBN; this draft does not classify it. **PDC-D3-R019** (PARTIAL; provider `DEFAULT_MODEL` constants, `LLMProfile.lab`, RAG default embedding provider; PDC finding F-D3-09, LOW) — the residual document assigns it to T-AF2-RW-RUNTIME and to this dossier; it must end fixed, N/A with evidence, or out of scope by Operator decision |
| S1 diagnostic (non-certifying, older baseline) | The S1 result reports ROADMAP-REPLAY-X `vendor_neutrality_replay_failures` = 1, a single `unexpected_prod_delta_for_vbn` violation caused by an unscoped product delta, and states the VBN artefact files and the PROHIBITED-unresolved count raise no violation. Quoted, not re-asserted; the counter must be re-proved at the FCS |
| Status @ `332c5217c` | Gates and register documents present; G6 ENT VBN 12/12 @ a scoped HEAD **≠** independent accept @ FREEZE_SHA |
| Acceptance @ FREEZE_SHA | `prohibited_unresolved=0` in frozen scope; every `PROHIBITED-*` in frozen scope `RESOLVED` or an accepted `ALLOWED-*` / ADR; no `EVIDENCE REQUIRED` left |

## §2.3 — Cross-layer replaceability matrix

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_replaceability_matrix.py`](../../../../tests/qualification/ebh_5/_replaceability_matrix.py) |
| Dual-slug / RPL-EF evidence | [`tests/qualification/ebh_5/_dual_slug_rpl_ef_matrix.py`](../../../../tests/qualification/ebh_5/_dual_slug_rpl_ef_matrix.py), [`test_ebh_5_dual_slug_rpl_e_operational_gates.py`](../../../../tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py) |
| ENT gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_replaceability_matrix.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_replaceability_matrix.py) |
| Rows @ draft SHA (read-only accessor call) | **56** matrix rows (fields `rpl_a`..`rpl_h`, `alternate_implementation_ref`, `behavioral_proof_ref`, `regression_gate_ids`): RPL-H `REPLACEABLE — QUALIFIED` 48 · `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE` 6 · `INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED` 2; undocumented qualified rows: 0 |
| Inventory vs matrix | 69 inventory records, 56 matrix rows: the difference of 13 equals the number of inventory records marked `extensibility_promised = no` (13 at the draft SHA); that the two sets coincide, concern by concern, is not shown here and is a W5/PT-Q13 replay item |
| Status @ `332c5217c` | Matrix SSOT tracked and consistent at the accessor level; certifying matrix replay @ FREEZE_SHA **OPEN** |
| Acceptance @ FREEZE_SHA | RPL-H rows without false `QUALIFIED`; matrix reconciled with inventory; every `extensibility_promised=yes` concern has RPL-A..G satisfied or an explicit `ARCHITECTURE BLOCKER` |

## §2.4 — Fixed substrate registry

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_fixed_substrate_registry.py`](../../../../tests/qualification/ebh_5/_fixed_substrate_registry.py) |
| RPL proofs | [`tests/qualification/ebh_5/_fixed_substrate_rpl_proofs.py`](../../../../tests/qualification/ebh_5/_fixed_substrate_rpl_proofs.py) |
| Decision records @ draft SHA (read-only accessor call) | **5** records; the registry's own validation returned 0 errors; every record's status is `RATIFIED — SCOPED QUALIFICATION EVIDENCE` (an Operator or PROD-Q scoped ratification, **not** an independent one) |
| Status @ `332c5217c` | Registry modules present; **independent** ratification sign-off @ FREEZE_SHA **OPEN** |
| Acceptance @ FREEZE_SHA | Fixed substrates **ratified only** (independently, per substrate); no silent bypass |

**Every fixed substrate below needs independent ratification at PT-VBN (T-B7-VBN-INDEPENDENT-ACCEPT).** The recorded ratification authority is quoted from the register; the draft adds no ratification of its own.

| Substrate id | Technology | Recorded authority (quoted) | Recorded multi-instance constraint (quoted) | Independent ratification |
| --- | --- | --- | --- | --- |
| `runtime-sqlite-persistence-bundle` | SQLite file-backed runtime stores (10 inventory concerns) | Operator 2026-10-28, OD-EBH5-04 runtime SQLite fixed substrate (scoped) | Single-node / profile-scoped; not horizontally scalable substrate | **REQUIRED — OPEN** |
| `single-node-profile-v1-sqlite-triad` | SQLite (GCA, PIS, agent distribution) | PROD-Q single-node profile qualification chain (scoped acceptance) | Strict single-node topology; advisory cross-process lock only | **REQUIRED — OPEN** |
| `lab-adaptive-sqlite` | SQLite adaptive signal/proposal stores | Operator 2026-10-28, OD-EBH5-04 lab adaptive SQLite (lab only) | Lab harness only; not production extensibility claim | **REQUIRED — OPEN** |
| `ebh7-conversational-memory-sqlite-lab` | SQLite conversational memory store (lab scope) | Operator 2026-10-31, EBH-7 OD-RPL-G2 | Lab/single-node conversational persistence; not full memory abstraction FIXED | **REQUIRED — OPEN** |
| `ebh7-integration-single-slug-vendor-slice` | Single catalog slug per integration category (5 inventory concerns) | Operator 2026-10-31, EBH-7 OD-RPL-G2 | Not vendor-equivalent; live sandbox replay required for Path B evidence | **REQUIRED — OPEN** |

**Fixed substrates the plan names that need a decision record or a guard (draft observations, not findings of record):**

- **SQLite single writer (EX-G-01 / PDC-D7-R012).** The residual document's bounded claim makes D7-R012 a §2.4 fixed substrate "(SQLite single writer) with a multi-instance fail-closed guard", and says the guard "must fail closed". Code that mentions single-writer files and multi-instance exclusivity exists (for example `intergrax/runtime/execution/continuation/durable_serialized_seal_store.py` and `intergrax/hosting/instance/file_guard.py`), but whether it satisfies the bounded claim for every SQLite touchpoint is **not verified here**. The substrate needs its own decision record naming the guard, then independent ratification.
- **In-memory graph defaults.** Plan section H lists "in-memory graph defaults" as a fixed substrate. None of the five records above names it. It needs a decision record (scope, guarantees, multi-instance implications, review policy) before it can be ratified.
- **Promised-extensible concerns classified as fixed or internal.** 56 inventory records have `extensibility_promised` of `yes` or `partial — WITH EVIDENCE`; 48 are classified `REPLACEABLE — QUALIFIED`. The other 8 are classified fixed or internal: `application.composition_root`, `memory.sqlite.conversational`, `prodq.colocated_runtime_adapter_isinstance` and the five integration categories `billing_meter`, `managed_retrieval`, `ml_inference_host`, `model_serving_runtime`, `vision_serving`. A promise with a fixed or internal classification is a point for the independent reviewer to confirm (promise withdrawn, or substrate ratified).

## §2.5 — Reachability / production-path proofs

| Field | Value |
| --- | --- |
| EBH-5 reachability gates | [`tests/qualification/ebh_5/test_ebh_5_reachability_gates.py`](../../../../tests/qualification/ebh_5/test_ebh_5_reachability_gates.py) |
| Runtime platform concerns | [`tests/qualification/ebh_5/_runtime_platform_concerns.py`](../../../../tests/qualification/ebh_5/_runtime_platform_concerns.py) |
| ENT gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_reachability.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_reachability.py) |
| Closed-world inputs (older baseline `7960ae7f6`, static, not re-measured here) | Sandbox providers E2B/Modal/Daytona: 24 of 25 provider modules reachable only through lazy imports from `intergrax/integrations/_shared/p7/factories.py` and `intergrax/integrations/registry/bootstrap_m6_p6.py`; no live run exists (PDC-stated). These providers are `NOT_CLAIMED` as production sandbox providers by OD-AF-18; the architecture admission and isolation gates are the claim. The closed-world result states that "reachable" is an over-approximation and "not reachable" is a strong static result |
| Status @ `332c5217c` | Reachability suites tracked; FREEZE-scope certifying run **OPEN** |
| Acceptance @ FREEZE_SHA | No shadow bypass of sanctioned composition boundaries |

## §2.6 — Behavioral qualification (non–monkeypatch-only)

| Field | Value |
| --- | --- |
| Behavioral registry | [`tests/qualification/ebh_5/_behavioral_proof_registry.py`](../../../../tests/qualification/ebh_5/_behavioral_proof_registry.py) |
| EBH-5 behavioral suites | [`test_ebh_5_behavioral_proofs.py`](../../../../tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py), [`test_ebh_5_behavioral_package_c.py`](../../../../tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py), [`test_ebh_5_behavioral_package_c_v2.py`](../../../../tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py) |
| ENT ops coupling | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_obs_ops_coupling.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_obs_ops_coupling.py) |
| Registry size @ draft SHA (read-only accessor call) | The qualified-concern behavioral proof registry holds **56** entries; 48 matrix rows are `REPLACEABLE — QUALIFIED` |
| Required level | **The full matrix, not a sample, at EBH-7 level** (plan section H step 1). The registry-to-matrix correspondence (56 entries against 48 qualified rows) is not explained by this draft and is a W5 reconciliation item; it must be explained, not assumed |
| Status @ `332c5217c` | Behavioral modules cited; independent behavioral attestation @ FREEZE_SHA **OPEN** |
| Acceptance @ FREEZE_SHA | Behavioral proofs cited for every extensibility claim; alternate injected through sanctioned composition; failures exercised without bypassing governance/execution/reliability contracts |

## §2.7 — Current-HEAD rollup @ FREEZE_CERTIFICATION_SHA

| Field | Value |
| --- | --- |
| This dossier | Tracked SSOT for ARCH-FREEZE VBN closure @ exact **FREEZE_CERTIFICATION_SHA** (draft until W5) |
| ENT VBN domain (**L2-VBN-ENT**) | `uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py -q` with `ENT_AUDIT_X_SEMANTIC=1` — **12/12** required @ FREEZE_SHA. At the draft SHA the acceptance folder holds 12 VBN modules (the 9 `test_ent_a_vbn_*` named in §2.1–2.6 and §3 plus the three `test_ent_a_ent_a_vbn_01..03`); wave-1 evidence reports 9 |
| SQLite effective gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_sqlite_effective.py`](../../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_sqlite_effective.py) |
| Engineering baseline | **T-B2-VBN** COMPLETE @ `f7ce9dae` (engineering ≠ independent acceptance) |
| Draft baseline | Tables §2.1–2.6 and §3 updated @ **`332c5217c`**; **FREEZE_CERTIFICATION_SHA** remains **TBD** |
| §2.7 targets | `prohibited_unresolved = 0` and `promised_unqualified = 0` at M3. At the draft SHA the inventory holds 0 `REPLACEABLE — NOT YET QUALIFIED` records; `prohibited_unresolved` is **not measured** here (it comes from the PT-Q13 certifying run) |
| Acceptance | Independent auditor sign-off; **FIND-VBN-PARTIAL** → **DISPOSITION** only after **T-B7-VBN-INDEPENDENT-ACCEPT** |

---

## §3 — SQLite mandatory annex (8 questions)

Schema: [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](./VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §3 (every production-effective SQLite touchpoint must answer the eight questions). Prior annex: [EBH_5_SQLITE_ANNEX.md](./EBH_5_SQLITE_ANNEX.md) (audit baseline `83a376b3`); the touchpoint register is [`_sqlite_touchpoint_registry.py`](../../../../tests/qualification/ebh_5/_sqlite_touchpoint_registry.py). **Working SQLite alone is not replaceability evidence.**

**Touchpoints @ draft SHA (read-only accessor call):** the register holds **39** touchpoints and the closed-world discovery of SQLite connect paths under `intergrax/` finds the same **39**, with no path on only one side. Classification: `ALLOWED — PROVIDER ADAPTER BOUNDARY` 5 · `ALLOWED — EXPLICITLY RATIFIED FIXED SUBSTRATE` 22 · `ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM` 12 · `REQUIRES ARCHITECTURE DECISION` 0. (Separately, 43 tracked non-test product files import `sqlite3`; that is a different measure from connect paths.)

| # | Question | Answer at the draft SHA | Open for the independent reviewer |
| --- | --- | --- | --- |
| Q1 | Intentional provider adapter? | 5 touchpoints are classified as provider adapter boundary; 22 as ratified fixed substrate and 12 as internal mechanism (register counts above) | Re-derive the classification of each of the 39 paths at M3 |
| Q2 | Canonical persistence contract exists? | The prior annex answers yes where the inventory record sets `canonical_contract_or_port`; the inventory validation returned 0 errors at the draft SHA | Confirm per touchpoint that the contract is the one consumers use |
| Q3 | Alternate compliant backend via sanctioned composition, without core or scenario change? | Prior annex: yes for the integration `relational_store` SQLite adapter (`REPLACEABLE — QUALIFIED`); **no** for runtime port implementations (fixed substrate) | Replay at M3 |
| Q4 | Core or business application change required to switch? | Prior annex: config-only for the integration adapter; adapter-only or BLOCKER for the fixed substrates | Replay at M3 |
| Q5 | Transaction, isolation, idempotency and recovery guarantees expressed without SQLite-only semantics? | Prior annex: expressed on the port contracts (STATE-X evidence); **not re-proved in this draft** | Independent check of the port contracts at M3 |
| Q6 | Configuration-driven? | Prior annex: yes for the integration adapter; fixed paths under the data directory for the single-node profile triad | Confirm no shadow selection (ENT `vbn_shadow_selection`) |
| Q7 | Explicitly ratified and documented if fixed? | 5 decision records exist (§2.4), all `RATIFIED — SCOPED QUALIFICATION EVIDENCE` by Operator or PROD-Q scope; **none is independent** | **Independent ratification of each substrate at PT-VBN**; decision records for SQLite single writer and for in-memory graph defaults (§2.4) |
| Q8 | Valid under **DIST-SCALE-X** multi-process / multi-worker claims? | **Valid only inside the recorded DIST-SCALE-X envelope.** Parent stage `DIST-SCALE-X` is recorded as `CLOSED — OPERATOR SCOPED ACCEPTANCE` with no independent audit on file ([parent reconciliation](./DIST_SCALE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md)); the WL-7 14400 s soak is `NOT PROVEN` and deliberately deferred ([record](./DIST_SCALE_X_WL7_4H_DEFERRED_NON_BLOCKING_RECORD.md)), the 3600 s substitute is the recorded evidence. The P0 scale matrix classes the SQLite runtime bundle **R** (representative-equivalence; "not multi-node proof") and PostgreSQL **M** for distributed durability ([matrix](./DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md)). Under DP-62 (as the residual document quotes it) the C10/C15 freeze is kept, DIST-SCALE-X envelopes are relabelled pre-D7, and DIST-SCALE-X stays BLOCKS until independent evidence exists (DP-42/62). The §2.4 SQLite claim is therefore **single-node, single writer**; any multi-process or multi-worker use is outside the claim unless a fail-closed multi-instance guard is shown | (a) name the guard and show that it fails closed on every SQLite touchpoint (EX-G-01 bounded claim); (b) tie the claim to the envelope version in force at the freeze; (c) no wording that implies multi-instance validity |

---

## Independent acceptance (mandatory)

| Role | Signature | Date | SHA |
| --- | --- | --- | --- |
| independent_auditor | **OPEN** | — | FREEZE_CERTIFICATION_SHA TBD |

**Hard gate:** **T-B7-PROMOTE-RPL** and ECAP §15 RPL-H rows require this sign block @ **FREEZE_CERTIFICATION_SHA** (not **T-B2-VBN** alone, and not this draft).
