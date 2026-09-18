# MP-6B — Collaborative Activity Core DTO & Contract Hardening

## 1. Scope

Hardening public contracts in `intergrax/contracts/collaborative_activity.py` and behavioral gates in `tests/unit/collaborative_work/test_mp6b_collaborative_activity_contracts.py`. No MP-6C+ runtime, no persistence providers, no source integrations. **MP-6A-C1-R1 append-store input remains `CollaborativeActivityPublication`** (frozen).

## 2. Git provenance

| Field | Value |
| --- | --- |
| SESSION_START_HEAD | `b7488cc72c6d6fcf32aff813b00e3e323059e169` (gate: `HEAD != origin/development` at session open) |
| MP6B_EVIDENCE_HEAD | `6ebc2b7909851228890d28a1ae3e6b0284b4b852` (immediately before MP-6B commit) |
| `origin/development` at evidence | `736ca764bb413fd661fcee8938dd0d4cae4f9404` |

## 3. Current contract inventory

| Contract | Semantic owner | Mutable? | Serialized? | Plugin-extensible? | Versioned? | Scope-bearing? |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CollaborativeActivityTypeId` | MP-6 identity | no | yes | yes | yes | no |
| `CollaborativeActivitySourceId` | MP-6 identity | no | yes | yes | yes | no |
| `CollaborativeActivityBuiltinType/Source` | platform taxonomy | n/a | via IDs | n/a | n/a | no |
| `ActivityIdempotencyKey` | MP-6 idempotency | no | yes | via type/source | yes | yes |
| `CollaborativeActivityActorRef` | MP-6 attribution | no | yes | kind enum | yes | tenant |
| `CollaborativeActivityScope` | MP-6 placement | no | yes | no | yes | yes |
| Target union members | MP-6 reference | no | yes | kinds fixed | yes | partial |
| Provenance union members | MP-6 reference | no | yes | kinds fixed | yes | partial |
| `CollaborativeActivityCorrelation` | optional correlation | no | yes | no | yes | no |
| `CollaborativeActivityOutcome` | MP-6 outcome | no | yes | status enum | yes | no |
| `CollaborativeActivityPublication` | producer command | no | yes | via key | yes | yes |
| `CollaborativeActivity` | materialized record | no | yes | via key | yes | yes |
| `CollaborativeActivityQuery` | read intent | no | yes | filter types | yes | yes |
| `CollaborativeActivityPageCursor` | pagination | no | yes | no | yes | no |
| `CollaborativeActivityPage` | read result | no | yes | no | yes | no |
| Ports / `CollaborativeActivityAppendStore` | SPI | n/a | no | impl-defined | n/a | n/a |

## 4. DTO classification

| Type | Classification |
| --- | --- |
| Type/Source IDs, idempotency key | IDENTITY VALUE OBJECT |
| Actor, scope, targets, provenance | SEMANTIC REFERENCE |
| `CollaborativeActivityPublication` | COMMAND / PUBLICATION DTO |
| `CollaborativeActivity` | MATERIALIZED RECORD |
| `CollaborativeActivityQuery` | QUERY DTO |
| Page cursor / page | PAGINATION / RESULT DTO |
| Ports | PORT / SPI CONTRACT |
| Hash helpers | INTERNAL HELPER |

## 5. Identity invariants

- `ActivityIdempotencyKey` is sole idempotency authority (`tenant_id`, `workspace_id`, `source`, `source_stable_id`, `activity_type`).
- `activity_id = mint_collaborative_activity_id(key)` — `activity-id/v1` length-prefixed SHA-256 (frozen).
- Golden vector: key in `test_mp6b_golden_activity_id` → `cact_35fa0b88b3369040c0378fc68db25a2f`.

## 6. Actor/scope invariants

- Actor `tenant_id` must match scope `tenant_id` on publication and materialized activity.
- Idempotency key tenant/workspace must match scope.
- Delegation fields paired; empty identity fields rejected (tests).

## 7. Target union

Kinds: `work_item`, `assignment`, `work_artifact`, `work_artifact_version`, `decision`, `approval`, `context_view`, `collaborative_decision_binding`, `collaborative_activity`. All `frozen=True`, `extra=forbid`, discriminator `kind`, no payload bodies.

## 8. Provenance union

Kinds: `execution`, `governance_evidence`, `proof_receipt`, `context_view`, `decision`, `approval`, `artifact_version`. Reuses `ExecutionProvenanceRef`, `GovernanceEvidenceRef`, `WorkArtifactVersionRef` where applicable. Duplicate locators: **deterministic canonical dedupe** via `model_dump_json` key (documented contract behavior).

## 9. Correlation

Optional only; not execution identity. **Fully empty `CollaborativeActivityCorrelation()` rejected** — use `correlation=null`.

## 10. Outcome

Statuses: `SUCCEEDED`, `FAILED`, `DENIED`, `PARTIAL`. `reason_code` optional, strip-normalized; no unstructured error payload.

## 11. Publication

No `activity_id`, `recorded_at`, `append_position`. `activity_type` derived read-only property from `idempotency_key.activity_type` (not serialized on wire).

## 12. Materialized activity

Defensive alignment: actor/scope/key/type/id, target↔scope, provenance artifact scope, correction semantics, `caused_by_activity_id != activity_id`. `append_position >= 1`; `recorded_at` timezone-aware.

## 13. Query/page

`tenant_id` + `workspace_id` required; `limit` 1–500; `occurred_after <= occurred_before`; `activity_types` deduped by qualified id; cursor opaque token. **Not an authorization proof.**

## 14. Time semantics

`occurred_at` = source event time; `recorded_at` = append materialization (store); `append_position` = workspace monotonic order (store); cursor = continuation — not conflated. No enforced `recorded_at >= occurred_at`.

## 15. Schema-version matrix

| DTO | Durable/public serialized? | `schema_version` present? | Decision |
| --- | ---: | ---: | --- |
| Core activity / publication / query / page | yes | yes | keep v1 literals |
| Target/provenance members | yes | yes | per-kind version |
| Type/source IDs | yes | yes | keep |
| Ports | no | n/a | n/a |
| Internal hash helpers | no | n/a | n/a |

## 16. Pluginability

`for_extension` type/source; custom port fakes in `test_mp6b_pluginability_custom_ports_satisfy_protocols`.

## 17. Reference-only guarantee

No generic dict payloads, evidence bodies, or trace bodies on MP-6 public DTOs (architecture AST gates + MP-6B tests).

## 18. Backend independence

No SQL/NoSQL/vendor imports in contract module (MP-6A gate).

## 19. Type-safety closure

`uv run pyright` on contract + MP-6A/C1/R1/B tests: **0 errors, 0 warnings**. `type: ignore` / `Any` / `cast` = 0 in contract.

## 20. Compatibility

| Change | Class |
| --- | --- |
| Empty correlation rejection | CONTROLLED HARDENING |
| `CollaborativeActivityPublicationPort.publish` body `...` | NON-BREAKING |
| Reject `CollaborativeActivityAppendIntent` at append store (WIP reverted) | NON-BREAKING (restores C1-R1) |

## 21. Regression

`uv run pytest` MP-6A + C1 + R1 + B modules: **64 passed**.

## 22. Final MP-6B verdict

**MP-6B — CLOSED / CERTIFIED** (subject to independent audit on GitHub). **MP-6C — NEXT.**
