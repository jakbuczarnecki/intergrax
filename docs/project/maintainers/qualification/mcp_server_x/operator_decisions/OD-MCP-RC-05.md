# OD-MCP-RC-05 — re-pin of the activation record: signature stands; F-49 and F-41 are fixed on `development` before the re-pin

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-RC-05 |
| **Status** | **RECORDED (operator chat answers, 2026-11-14; not signed `Authorized:` lines)** |
| **Related** | `OD-MCP-RC-04` sections 5 and 6, `PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_2.md` (section 5.5), `PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_3.md` (F-49, F-41) |

English renderings below are by the orchestrator session and are not operator-written lines; the Polish option labels are quoted as selected.

## 1. The signature of 2026-11-14T08:21:13+01:00 stands for the re-pin

Question: the re-pin moves the audited tree from `324d9dba3` to a later head; the signed decision file compares no SHA in the gate, but part of its text becomes stale. Does the existing signature stand for the re-pin. Answer: "Tak, podpis obowiązuje (Recommended)" ("Yes, the signature stands"). Effect: the signed file `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` stays byte-identical (SHA-256 `0e0aed5816ff76c631d8aefb897e83399d26aa6d9692f3b0c2fe6b1fd8d25147`). The activation certifier lists the statements of the signed text that the re-pin makes stale (audited tree, "no product-path change on `development` after `a156cbbd5`", the audit table ending at addendum 2, open conditions 2 and 8) in the `limitations` of the new binding manifest and in the handoff record. No new signature, no delegation renewal.

## 2. F-49: the `openai` row is repaired before the re-pin

Question: the sibling shared-slug row `openai` (categories managed_retrieval and embedding_provider) has the defect that T-IGX-036-X repaired for `ollama`; repairing it is a new product path while `development` is frozen. Answer: "Napraw przed rebindem" ("Fix before the rebind"). Effect: this is a product change owned by PDC-X on `development`, a deliberate exception to the freeze, covered by a new PDC-X ticket (not MCP-RC). It adds a pinned test for the row (and the `enabled` / keyword pass-through tests of F-50 may ride along as test-only additions).

## 3. F-41: the impact-map rule is added before the re-pin

Question: the eight `applications/*/mcp/server.py` modules map to no check in the ENT-AUDIT impact map; `xd_12` and `p6_impact` tests are red on `mcp-rc`. Answer: "Reguła przed rebindem" ("Rule before the rebind"). Effect: the PDC-X / ENT-AUDIT owner adds a rule for the `applications/*` roots to `intergrax/platform/enterprise_audit/impact_map.py` and `changed_since_impact_map.v1.json` on `development` without narrowing either test. This voids audit addendum 2 for those two paths.

## Consequence

Both changes are product-path changes on `development`. When PDC-X states that they have landed and `development` is frozen again, the sequence repeats once: merge `development` into `mcp-rc`; the independent auditor writes addendum 4 for every product path changed since `0fc8badd8` (the `openai` row, the two impact-map paths, and anything else the gate lists); the evidence integration round 5; the certifier preflight addendum 3; the activation certifier re-pins the record in one new two-phase update (corrective and audit pins at the new audited tree, new binding manifest and handoff); only then the PR, the five CI contexts on the pinned head, the merge and the terminal verification. MCP-RC does not edit `intergrax/integrations/` or `intergrax/platform/enterprise_audit/` (outside its ticket write sets).

## Explicit non-actions

No signature, no rebind, no merge, no product change by MCP-RC, no FRZ change (counter stays 225).
