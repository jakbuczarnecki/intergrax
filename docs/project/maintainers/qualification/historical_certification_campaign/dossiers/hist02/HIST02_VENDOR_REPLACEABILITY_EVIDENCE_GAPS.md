# HIST-02 Vendor / Replaceability Evidence Gaps

HEAD `ba07ebaa34d04633bd60510f68c18346b1ef160d`. Static inventory; nothing executed.

## A. Hardcoding inventories (reused, not duplicated)
- EBH-6 CORE_HARDCODING_FINDINGS_REGISTER (57 findings: 0 unclassified, 0 in-scope PROHIBITED-* unresolved at its qualification SHA; layer attestation CANDIDATE-RESOLVED, independent resolution via Protocol 03).
- ARCH-FREEZE B1 closed-world inventory: 9024 modules classified, strict_pass @ 6bdf78c, 12 production_effective, 21 `unknown:dynamic` nodes dispositioned EXTRACTOR_LIMITATION (arch_freeze_b1_mandatory_dispositions.json).
- Gap: both are static; dynamic imports (incl. intergrax/cli/external_work.py importlib sites) are not behaviorally proven (F-002).

## B. Replaceability vs actual execution surfaces
- EBH-5 proofs register real alternate factories; monkeypatch only sets env slug variables (F-010). Execution surfaces proven: integration catalog resolution, LLM adapters, OTLP/collector transport (W5), analyzer port (W6), Kv/DocumentStore adapters (TRACE-X-P5-R2-P2).
- Not shown by any HIST-02-visible record: replaceability of the adapter registries listed in F-005 through the Integration contract path.
- Shadow/fallback: `graph_store_backend` gone from code; no RAG-local provider registry found by grep. Parallel registries in F-005 need disposition.

## C. GAP-RPL-51 and EBH-7 supersession
- EBH_7_GAP_REGISTER: GAP-RPL-51 CLOSED, 56/56 QUALIFIED @ HEAD (W5).
- EBH_5 records mark the earlier 51 NYQ count as superseded by EBH-7 (historical text intact).
- EBH_7_RPL_H_BATCH_DISPOSITION_RECORD: exact split QUALIFIED / FIXED SUBSTRATE / INTERNAL is deferred to T-B7-VBN-INDEPENDENT-ACCEPT.
- Owner and gate: independent_auditor, BLOCKED_EXTERNAL, deps T-B7-VBN-HEAD-DOSSIER + T-B7-LOCK-FREEZE-SHA (WBS L40). Classification for RPL-01/02/04: INDEPENDENT_AUDIT_GAP; RPL-03: EVIDENCE_GAP with INDEPENDENT_AUDIT_GAP secondary. Not closed or implemented here.

## D. Row-level gaps
- RPL-01/02/04: auditor replay of ENT VBN checks @ FREEZE_SHA.
- RPL-03: no explicit not-monkeypatch-only gate.
- PLG-02: external-plugin gates exist, unmapped.
- CTR-02/06: EBH-6 hardcoding register gate replay + independent L9.
- PLG-08/OWN-04/05/TYP-05/PLG-06/07: see findings.
