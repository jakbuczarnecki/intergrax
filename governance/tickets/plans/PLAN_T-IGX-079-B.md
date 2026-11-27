# T-IGX-079-B — INTELLIGENCE-X R2 documentation candidate adoption (proposed)

## Scope and authority

Adopt the exact three R2 document candidates after independent document review and formal plan approval. Reference review baseline `7d2cc3d28640df4823fa04e07abdc8f98451ccc6`; original preparation baseline `32005c212ed4debca91f607ee30b3abfab79ce6c`. This is a proposed limited adoption batch, not the previously withdrawn II-W0-RATIFICATION-01 and not full W0 closure. T-IGX-079-B was unallocated at the review baseline; recheck current repository and resolve collisions before adoption. No implicit approval or backdated signature.

## Exact payload and write-set

| Path | Normalized plan SHA-256 |
|---|---|
| `docs/project/capabilities/architecture/INTEGRAX_INTELLIGENCE.md` | `c9de0752dbb45aef0c8b9de584ee308eecf9da44aba09928af6ac051c107130a` |
| `governance/tickets/plans/INTELLIGENCE-X-MASTER-PLAN.md` | `fbc8d2857b1da8967395ca3681413a2f732bfb0b549d0cf6ca4256b59622f895` |
| `docs/project/capabilities/plan/INTEGRAX_INTELLIGENCE.md` | `01b561bb92d3868ef80fcb77e1898450186275a86d0308f9a6910b988b5d9d7a` |

Only those three candidate paths plus `governance/tickets/T-IGX-079-B.md` and `governance/tickets/plans/PLAN_T-IGX-079-B.md` if this proposed ID is confirmed. No additional tracked paths. Evidence files, patches and validation JSON stay outside the tracked repository (approved evidence location or `.tmp/`). Canonical architecture and roadmap remain clearly marked candidate until a separately authorized ratification decision. Preserve all prior approvals and evidence.

## Preconditions and procedure

1. Operator or eligible separately commissioned delegate approves the exact plan digest in the actual ticket after review; implementer never signs. Verify role authority and intended isolated worktree. Record fresh development SHA and ancestry; require architecture blob `3ffd8de71737a3b606a72e89e439854eabaf9ef2`, plan index blob `71329af63765c5c4e671880a8921de4103c4b6f2` and master plan absent, or stop for exact reconciliation. Material source drift affecting requirements requires new review. Preserve unrelated work and protected paths.
2. Apply supplied `proposed.diff` in isolated authorized workspace only after approval. Recompute the three digests with existing `scripts/governance/plan_digest.py`. Verify byte equality or normalized digests plus expected line endings; if content differs, stop. Verify write-set and zero product/held-path delta. Do not change global roadmap, SFX/SEF plans, FRZ, other domain docs, governance roles, scripts, tests or implementation.
3. Run the existing plan-approval verifier, diff check, applicable existing documentation/link checks, and governance ticket smoke sequentially; do not install new dependencies or weaken checks. Use an existing permitted Mermaid parser if available; otherwise explicitly record parser not executed and have reviewer disposition the limitation. Record command, exit result, exact candidate SHA and limitations.
4. Independent reviewer checks exact candidate, original requirement preservation, F01–F08, source consistency and procedural order. Implementer cannot self-certify. Record findings and resolve material defects before acceptance; changed content needs delta review and renewed approval when the plan digest changes. Authorized publisher alone performs commit/publication/integration per canonical role policy. Record remote evidence and verify final exact tree. No W0 package formally closed merely by adoption. Scope-specific formal ratification and W0-06 protected amendments remain separate.

## Test contract

- `uv run python scripts/governance/verify_plan_approval.py governance/tickets/T-IGX-079-B.md --plan governance/tickets/plans/PLAN_T-IGX-079-B.md` (confirm actual CLI at execution baseline).
- `git apply --check` before application; `git diff --check`; exact write-set and zero product delta; compare all normalized digests using `scripts/governance/plan_digest.py`; actual Markdown link and DAG validation; existing doc/diagram validator if available and authorized.
- `uv run python -m scripts.governance.igx_gov ticket --profile smoke` (sequential, existing hooks unchanged).

## Rollback and exclusions

Use authorized reviewed revert of this isolated document delta only; never reset shared branches or revert foreign changes. No product implementation, executable W1, new runtime, gate opening, FRZ/IIAC promotion, global certification, SFX/SEF master-plan edits or protected roadmap changes. This draft grants no authority.
