# HIST-02 Architecture Invariant Audit (35 rows)

Audited HEAD `ba07ebaa34d04633bd60510f68c18346b1ef160d` (= CAMPAIGN_START_SHA), branch hist-cert/hist02-architecture. Generated 2026-11-08T07:24:48Z.
Method: static inventory only (git/grep/Read). No tests executed. No row promoted; all 35 stay OPEN.

## 1. Baseline
All 35 rows (BND 6, OWN 5, CTR 6, TYP 6, PLG 8, RPL 4) are OPEN in PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md L78-114. No divergence from the brief.

## 2. Canonical sources used (no new authority created)
- Checklist; QUAL-X-P5 freeze selection manifest; QUAL-X-FINAL invariant coverage catalog; EBH-6 family L9 evidence bundles; ROADMAP-REPLAY-X rows; ARCH-FREEZE WBS/DAG, B1 inventory and dispositions; EBH-5/6/7 qualification records.

## 3. Results by classification (primary, one per ID)
- AUTHORITY_GAP: 1
- EVIDENCE_GAP: 15
- INDEPENDENT_AUDIT_GAP: 3
- QUALIFICATION_GAP: 7
- SSOT_DRIFT: 1
- VALID_SCOPED_PASS_CANDIDATE: 8

Dashboard states:
- AUDIT_REQUIRED: 12
- AUTHORITY_REQUIRED: 1
- QUALIFICATION_REQUIRED: 22

## 4. Per-ID table
| FRZ | Primary | State | Secondary | Owner of remaining gate | Remaining gate |
|---|---|---|---|---|---|
| FRZ-BND-01 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | - | ARCH-FREEZE B7 (T-B7-PROMOTE-BND) | P5-F01 current-HEAD replay at pinned SHA + independent audit @ FREEZE_SHA + operator/auditor promotion |
| FRZ-BND-02 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | QUALIFICATION_GAP | HIST-02 (mapping) -> ARCH-FREEZE B7 | Map explicit direction rules (AGENTS.md hard boundaries) to hardening_3 gate as per-ID evidence; replay; audit |
| FRZ-BND-03 | AUTHORITY_GAP | AUTHORITY_REQUIRED | SSOT_DRIFT, EVIDENCE_GAP | Operator (exception) / ARCH-FREEZE B7 | Operator decision on Tier-0->Tier-3 exception in intergrax/cli/external_work.py (accept as limitation with decision id, or relocate to Tier-3) + dynamic-edge coverage; then replay and audit |
| FRZ-BND-04 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping of EBH-2A/EBH-3 gates; replay; audit |
| FRZ-BND-05 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | - | ARCH-FREEZE B7 | Replay of EBH-6 BND bundle gate at pinned SHA; audit; promotion |
| FRZ-BND-06 | SSOT_DRIFT | QUALIFICATION_REQUIRED | QUALIFICATION_GAP | HIST-04 (SSOT) / ARCH-FREEZE B7 | Reconcile checklist cell ('-') with QUAL-X-FINAL/P5 F01 proof; extend proof beyond OTLP/collector transports (hosting/queue/MCP/HTTP) or record scope limitation |
| FRZ-OWN-01 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | INDEPENDENT_AUDIT_GAP | ARCH-FREEZE B7 | Platform-wide owner uniqueness gate (P5 F02 = PARTIAL_PLATFORM_GAP, 'DUP-X certification not claimed'); replay; audit |
| FRZ-OWN-02 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping + replay + audit |
| FRZ-OWN-03 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping + replay + audit |
| FRZ-OWN-04 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | EVIDENCE_GAP | HIST-02 / ARCH-FREEZE B7 | Closed-world shadow-registry gate or recorded disposition for vision/speech/document-splitter adapter registries; replay; audit |
| FRZ-OWN-05 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | EVIDENCE_GAP | HIST-02 / ARCH-FREEZE B7 | Same as OWN-04 (shared cause F-005); replay; audit |
| FRZ-CTR-01 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | - | ARCH-FREEZE B7 | P5 F03 replay + audit |
| FRZ-CTR-02 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | INDEPENDENT_AUDIT_GAP | ARCH-FREEZE B7 / VBN | EBH-6 hardcoding register gate replay + ARCH-FREEZE B1 inventory reuse + independent L9 (VBN bundle) |
| FRZ-CTR-03 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID platform-wide dict-contract scan or scoped limitation; replay; audit |
| FRZ-CTR-04 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping (EBH-2A public-contract gate candidate); replay; audit |
| FRZ-CTR-05 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping + replay + audit |
| FRZ-CTR-06 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | INDEPENDENT_AUDIT_GAP | ARCH-FREEZE B7 / VBN | Same as CTR-02 |
| FRZ-TYP-01 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | - | ARCH-FREEZE B7 | P5 F03 replay + pyright gates; audit |
| FRZ-TYP-02 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping; AST Any-scan replay; audit |
| FRZ-TYP-03 | EVIDENCE_GAP | AUDIT_REQUIRED | ACCEPTED_LIMITATION | ARCH-FREEZE B7 / auditor | Auditor judgment on nexus_loop.py:680,1037 AuthoritativeDecisionExposure[object]; replay |
| FRZ-TYP-04 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | EVIDENCE_GAP | ARCH-FREEZE B7 | Dynamic-dispatch inventory beyond static gates: 21 B1 'unknown:dynamic' nodes (EXTRACTOR_LIMITATION) and cli/external_work.py importlib imports |
| FRZ-TYP-05 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Platform-wide string-dispatch gate or scoped limitation; replay; audit |
| FRZ-TYP-06 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Per-ID mapping; cast/type-ignore AST scan replay; audit |
| FRZ-PLG-01 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | - | ARCH-FREEZE B7 | P5 F03 replay + audit |
| FRZ-PLG-02 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | INDEPENDENT_AUDIT_GAP | ARCH-FREEZE B7 | Map test_external_plugin/EBH-5 behavioral proofs; replay; audit |
| FRZ-PLG-03 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Platform-wide discovery owner statement + gate mapping; replay; audit |
| FRZ-PLG-04 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | - | ARCH-FREEZE B7 | EBH-6 PLG bundle replay; audit |
| FRZ-PLG-05 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Map catalog hot-reload governance gates (CHR1-12); replay; audit |
| FRZ-PLG-06 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | SSOT_DRIFT | ARCH-FREEZE B7 | Platform-wide gate for tool/skill/MCP/integration packages; fix citation G11-17 |
| FRZ-PLG-07 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | - | ARCH-FREEZE B7 | Platform-wide extension-authority gate; replay; audit |
| FRZ-PLG-08 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | QUALIFICATION_GAP | HIST-02 (mapping) -> ARCH-FREEZE B7 | Map catalog/tool/skill/context registry reversibility tests to the ID; confirm uniform scope across registries; replay; audit |
| FRZ-RPL-01 | INDEPENDENT_AUDIT_GAP | AUDIT_REQUIRED | SSOT_DRIFT | independent_auditor (T-B7-VBN-INDEPENDENT-ACCEPT, BLOCKED_EXTERNAL) | T-B7-VBN-HEAD-DOSSIER -> independent replay @ FREEZE_SHA -> T-B7-PROMOTE-RPL |
| FRZ-RPL-02 | INDEPENDENT_AUDIT_GAP | AUDIT_REQUIRED | SSOT_DRIFT | independent_auditor (T-B7-VBN-INDEPENDENT-ACCEPT) | as RPL-01 |
| FRZ-RPL-03 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | INDEPENDENT_AUDIT_GAP | HIST-02 (gate) + independent_auditor | Explicit gate that EBH-5/7 proofs register real alternate factories and use monkeypatch only for env selection; then T-B7-VBN-INDEPENDENT-ACCEPT |
| FRZ-RPL-04 | INDEPENDENT_AUDIT_GAP | AUDIT_REQUIRED | SSOT_DRIFT | independent_auditor (T-B7-VBN-INDEPENDENT-ACCEPT) | as RPL-01 |

Full per-ID evidence (SSOT selections, EBH-6 bundles, candidate gates, existence checks, rationale): HIST02_FRZ_35_PER_ID_MATRIX.json.

## 5. Inherited scoped proofs
Every checklist evidence cell is scoped ('parent scope; not global PASS'). Accepted SHAs (EBH-5 33ecf19cc, EBH-6 f27bfaf2f, EBH-7 dc18e36d9/df498b7a8, EBH-2G 33175f8e, EBH-2F 765fabe38, CE-01 a8d47a639, AW-7C-P1-ARCH fa5853ec0, INT-CONFIG a59744517, S24-GAP-04-P2-R2 041057266, PROD-Q-P2 8e43b1604) all exist and are ancestors of HEAD, 456 to 1901 commits behind. None is presented as current-HEAD platform proof.

## 6. Post-acceptance drift (governed paths)
Since EBH-7 baseline dc18e36d9: intergrax/ 13 commits, 34 files, all ENT-AUDIT-X (out of scope, dependency only); agents/ and applications/ unchanged; tests/qualification/ebh_5 unchanged; ebh_7 tests 6 files (ENT-AUDIT-X sync). The new intergrax/platform package has no imports toward runtime/agents/applications. Drift older than EBH-7 is covered by EBH-6 (f27bfaf2f) and EBH-7 as independently accepted scoped certificates.

## 7. Cross-layer ownership and authority
- Static scan: agents/ -> applications/: 0. intergrax/ -> applications/ or agents/: 1 (F-001), plus dynamic edges (F-002).
- graph_store_backend: absent from code. Provider selection/discovery owner = Integration catalog/IntegrationProfile per scoped EBH-2G evidence (not re-executed).
- Shadow registries / parallel mechanisms: 3 module-level adapter registries beside the Integration catalog, undispositioned in HIST-02 scope (F-005).

## 8. Specific blockers (not prose gaps)
See HIST02_FINDINGS_AND_RESIDUALS.json: F-001 (BND-03 authority), F-004 (PLG-08 unmapped), F-005 (OWN-04/05), F-007 (RPL-01..04 auditor gate, BLOCKER), F-008/F-009 (PLG-06/07/TYP-05 single-scope), F-011 (28 rows without a replay owner).

## 9. Special-scrutiny rows
- PLG-08: implementation and tests exist, evidence unmapped -> EVIDENCE_GAP, not PASS.
- RPL-01..04: auditor-owned gate (BLOCKED_EXTERNAL). EBH-5/EBH-7 closure not used to infer PASS. GAP-RPL-51 register/record tension noted (F-007).

## 10. Not done / out of scope
No ARCH-FREEZE R1-R6 work, no ENT-AUDIT-X touch, no FRZ promotion, no code change, no test execution.
