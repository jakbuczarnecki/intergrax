# T-IGX-046-D — EQO-A remediation node PT-D08 (batch DB-H): documentation under product-path prefixes (agents roster, Neo4j graph-store usage)

## Scope

Product-path documentation ticket of the integrated wave (master-plan node `PT-D08`; RW-PLAN v13 appendix A1 block `PT-D08 (T-IGX-046-D)`), from the Campaign A dossier findings EQOA-A4-041 and EQOA-A2-012 (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589), re-read at `origin/development` and `origin/arch-freeze/rw-train` (both files unchanged since the baseline). Four hunks in two files: in `agents/README.md` five roster rows, three corrected lifecycle values of existing rows and one label line under the roster table (the lifecycle column is a contract state, not a certification statement); in `intergrax/integrations/providers/graph_store/neo4j/USAGE.md` the replacement of a stale graph-store enablement sentence. Both files are product paths under `PRODUCT_PATH_PREFIXES`; no code, test or script changes. No PASS, closure, FRZ promotion, stage certification or independent-verification claim.

ACT-N11: any change to these paths breaks the IIAC activation until the single RB-2 re-pin. This ticket does not perform the re-pin; it rides in the wave train (pull request into `arch-freeze/rw-train`, lane 2 after PT-WA and PT-WB) and must not merge into `development` outside the train. Its approval line goes through a wave packet (OD-AF-22), because it touches product-path documents; the signing delegate alone does not approve it.

## Plan digest

Plan digest: f0dc7bdcda2a1d6df3342eb1881b89e799edd44bb626231f4ebd730e593e1dc2

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-046-D.md`. The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
agents/README.md
intergrax/integrations/providers/graph_store/neo4j/USAGE.md
```

## Approved

Approved: operator-delegate @ 2026-11-18T09:48:29+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_STANDING_DELEGATION_RECORD.md@881428fb231f34215a67364aa5788a4f2f986e83
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Implementer

Implementer session: local_99c37654-4b83-40fb-a4fe-c0d8547a792f

The independent review of this ticket must be done by a session that is not the implementer.

## Test contract

One process at a time, under the shared pytest lock (`D:/Projekty/_wt/_locks/pytest.lock`, at most 15 minutes):

- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer check: the pull request lists exactly the two documents of the write set plus this ticket and its plan.

## Out of scope

`intergrax/tools/providers/rag/USAGE.md` and `docs/project/technical/guides/RAG_OPERATOR_GUIDE.md` (other stale selector mentions), every code, test, script, workflow, roadmap, FRZ checklist, PDC-X and IIAC record change, the IIAC re-pin (RB-2), GitHub settings, Docker, vendor work.

## Lifecycle state

plan_approved
