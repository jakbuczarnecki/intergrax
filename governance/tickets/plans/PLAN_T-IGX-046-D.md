# PLAN T-IGX-046-D — EQO-A remediation node PT-D08 (batch DB-H): documentation under product-path prefixes (agents roster, Neo4j graph-store usage)

Approval state: see the Approved section of the ticket `governance/tickets/T-IGX-046-D.md`. Approval of this plan is not an implementation go and not an audit. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no code.

## 1. Purpose

Correct two documentation statements that the EQO-A Campaign A baseline audit (dossier PR #589, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`) found stale, in the two documents that sit under the product-path prefixes of `scripts/governance/iiac_common.py` (`PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")`):

- EQOA-A4-041: the agent roster in `agents/README.md` lacks five agent packages that exist and are imported by host applications; its lifecycle column carries no label saying what the values are, and three existing rows disagree with the `lifecycle_state` their agent contract declares.
- EQOA-A2-012: `intergrax/integrations/providers/graph_store/neo4j/USAGE.md` still names `INTERGRAX_RAG_GRAPH_STORE` and `RagProfile.graph_store_backend` as the way to enable the RAG graph backend; both are gone from the RAG composition (EBH-2G-R2-R1).

The Operator's recorded answers of 2026-11-16 name this node as a documentation batch that touches product paths and rides in the single integrated wave (DP-10, DP-11; RW-PLAN v13 appendix A1 block `PT-D08 (T-IGX-046-D)`).

## 2. Product-path and train handling (ACT-N11)

- Both files are product paths. Any change to them after the IIAC activation binding breaks the activation (ACT-N11) until the single re-pin (RB-2) at the final product tree. This ticket does not perform the re-pin and must not merge into `development` outside the wave train.
- The change rides in the wave train: its pull request targets `arch-freeze/rw-train` (lane 2, after PT-WA and PT-WB have merged into the train, RW-PLAN section (c)); the merge of the train into `development` stays the Operator's decision.
- Approval route: the approval line of this ticket goes through a wave packet (OD-AF-22), because the ticket touches product-path documents; the signing delegate alone does not approve it. The packet record is prepared by the orchestrator; this plan does not write it.
- The two files are inside the scope of the wave audit (node PT-Q02) even though the change is documentation only.

## 3. Scope (4 hunks, 2 files)

| Hunk | File | Change |
|---|---|---|
| DB-H-H01 | `agents/README.md` | Five rows appended to the "Agent roster" table directly before the `Lab mock agents` row: `BoundaryDemoAgent` (`attestation.demo`, staging, `attestation_demo`, `boundary_demo/`), `ExternalContractorAdapterAgent` (`external_contractor.adapt`, production, `governed_contractor_application`, `external_contractor_adapter/`), `ModelRoutingQualifierAgent` (`local.workspace.model_routing_qualification`, staging, `local_workspace_application`, `model_routing_qualifier/`), `ToolSelectionQualifierAgent` (`local.workspace.tool_selection_qualification`, staging, `local_workspace_application`, `tool_selection_qualifier/`), `WebSearchQualifierAgent` (`local.workspace.web_search_qualification`, staging, `local_workspace_application`, `web_search_qualifier/`). The lifecycle values of the new rows are the `lifecycle_state` of each agent's `build_agent_contract()`. |
| DB-H-H03 | `agents/README.md` | Three existing roster rows are corrected to the `lifecycle_state` their contract declares: `SignoffProbeAgent` `staging` becomes `development` (`agents/signoff_probe/contract.py` L22) and `OrganizationWorkerAgent` `development` becomes `staging` (`agents/organization_worker/organization_worker_agent.py` L91, in `get_contract`), and `Lab mock agents` `-` becomes `development` (`agents/lab/contract.py` L35, `_mock_contract`, used for all four mock agents through `build_agent_contract`). |
| DB-H-H04 | `agents/README.md` | One label line is added under the roster table, separated by blank lines from the `Lab mock agents` row and from the `---` rule: "Lifecycle = the `lifecycle_state` declared in each agent's `AgentContract`. It is a contract state, not a certification, audit or readiness result." |
| DB-H-H02 | `intergrax/integrations/providers/graph_store/neo4j/USAGE.md` | The two-line sentence "Enable the RAG graph backend with `INTERGRAX_RAG_GRAPH_STORE=neo4j` (or `RagProfile.graph_store_backend`)." is replaced by a paragraph that names the Integration profile path: bind `IntegrationProfile(graph_store=IntegrationBinding.from_slug("neo4j"))`, pass the profile to `create_default_rag_stack(...)`, which resolves the `GRAPH_STORE` category through `resolve_from_profile`; set `RagProfile.graph_rag_enabled=True` to use GraphRAG; state that `RagProfile` has no `graph_store_backend` field and that the RAG composition modules do not read `INTERGRAX_RAG_GRAPH_STORE` (both removed by EBH-2G-R2-R1; gate `tests/unit/rag/graph/test_ebh_2g_r2_graph_store_ownership_gate.py`). |

Nothing else in either file changes. The exact text is the unified diff `PATCH_DB-H_amend2.diff` of the drafting record (directory `.tmp/session/arch-freeze/w3-packet-2/046-D/`); `git apply --check` of it is clean against the train head `881428fb231f34215a67364aa5788a4f2f986e83`.

## 4. Facts and sources

Every statement is read at the baseline and re-read at the train head (`origin/arch-freeze/rw-train`) and at `origin/development`: class names, capability ids and `lifecycle_state` of the five agents (`agents/<slug>/contract.py`, `capabilities.py`), the host imports (`applications/attestation_demo/manifest.py`, `applications/governed_contractor_application/manifest.py`, `applications/local_workspace_application/manifest.py`), the absence of `graph_store_backend` and `INTERGRAX_RAG_GRAPH_STORE` from `intergrax/`, `agents/`, `applications/` code, `RagProfile.graph_rag_enabled` (`intergrax/rag/profiles/rag_profile.py`), `resolve_from_profile` and `create_default_rag_stack`, and the ownership gate test. The `lifecycle_state` of every agent in the roster (every `lifecycle_state=` declaration under `agents/`, including the organization worker's `get_contract`) was re-read at `881428fb231f34215a67364aa5788a4f2f986e83`; after the three corrections of DB-H-H03 every roster row (22 rows) agrees with its contract, checked row by row against the declarations. The drafting record is `.tmp/session/eqo-a-remediation/WP-9/patches/DB-H/FACT_CHECK.md`.

## 5. Explicitly out of scope

- `intergrax/tools/providers/rag/USAGE.md` L63 (also a product-path document) and `docs/project/technical/guides/RAG_OPERATOR_GUIDE.md` L121 and L140 still list `INTERGRAX_RAG_GRAPH_STORE` as a selector. They are not in RW-PLAN appendix A1 for this ticket; adding the first one would be an amendment (3 files).
- Any code, test, script, workflow, roadmap, FRZ checklist, PDC-X or IIAC record, GitHub setting; the IIAC re-pin (RB-2); any certification statement: no sentence asserts or upgrades independent verification, PASS or closure.

## 6. Write set

```text
agents/README.md
intergrax/integrations/providers/graph_store/neo4j/USAGE.md
governance/tickets/T-IGX-046-D.md
governance/tickets/plans/PLAN_T-IGX-046-D.md
```

Product paths: 2 firm, 0 new, 0 conditional. Tests: none. Negative tests: none (documentation patch, RW-PLAN section (d)).

## 7. Test contract

One process at a time, under the shared pytest lock:

- `uv run python scripts/docs/check_docs_domain_pairs.py` (RW-PLAN section (d), docs pair check).
- Reviewer checks: the pull request lists exactly the two documents plus this ticket and its plan; every added statement matches the sources of section 4 at the train head; no line of either document changes beyond the four hunks of section 3.

## 8. Risks

- ACT-N11: see section 2. Merging the pull request anywhere but into `arch-freeze/rw-train` breaks the IIAC activation until the re-pin.
- Roster drift: PT-WA changes the constructor of `WebSearchQualifierAgent` and deletes `search_provider_resolver.py`; the roster columns do not depend on either, but the rows must be re-checked if an agent's lifecycle, capability id or host changes before this ticket merges.
- A second stale selector mention remains outside the write set (section 5); its follow-up sentence for the ARCH-FREEZE register is: "`intergrax/tools/providers/rag/USAGE.md` L63 (a product-path document) and `docs/project/technical/guides/RAG_OPERATOR_GUIDE.md` L121 and L140 still list `INTERGRAX_RAG_GRAPH_STORE` as a graph-store selector although the RAG composition no longer reads it (gate `test_ebh_2g_r2_graph_store_ownership_gate.py`); correct both in a later documentation ticket, the first one inside the wave train."

## 9. Rollback

Revert the single pull request. No data, configuration or code path is affected.

## 10. Closure condition

Pull request merged into the train after independent review; the ticket records the merge SHA. No finding is marked closed or verified by this ticket.
