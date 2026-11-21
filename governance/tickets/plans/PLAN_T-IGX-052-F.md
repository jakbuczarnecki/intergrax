# PLAN T-IGX-052-F — W6 Campaign B omega prep (governance refinement)

Status: **DRAFT** for Operator review. Source: MEP-AF2 v2.1 W6 ladder; WP-12 nodes `PT-B1W`, `PT-B2W`, `PT-BA`, `PT-D13`. Facts read at branch base `e69e1ff58` (`origin/development` ancestry). Nothing here approves, executes or certifies a campaign.

## 1. Purpose

W6 prep documentation (`W6_CAMPAIGN_PREP_CHECKLIST.md`, `W6_PREP_DOSSIER.md`) already names campaign nodes and prep rows. This batch **refines only the delivery ticket and plan** so implementer and auditor sessions have an English SSOT with an explicit write set, prerequisites, and verification contract before any Campaign B omega execution batch. It claims no Campaign B PASS, no independent auditor session, no omega @ FCS evidence, and no FRZ promotion.

## 2. Prerequisites (program — not met @ `e69e1ff58`)

| Id | Prerequisite | Notes |
|----|--------------|-------|
| F-P0 | Rung **FCS**; product tree ≡ G2 | **BLOCKED** @ base |
| F-P1 | B-1/B-2 alpha complete on `arch-freeze/rw-train` (W2/W3) | MEP W6 row |
| F-P2 | W5 pre-pin artefacts merged (manifests, ECAP, VBN finals per MEP) | T-IGX-052-A..D reserved |
| F-P3 | AR-57 commission for `PT-BA`; AR-08/AR-10/AR-53 cited for B omega slots | authorization **not** executed |

## 3. Node map (execution deferred)

| Node | Role | Prep checklist rows (read-only) |
|------|------|--------------------------------|
| `PT-B1W` | Campaign B-1 omega @ FCS | W6C-A1..A4 |
| `PT-B2W` | Campaign B-2 omega @ FCS | W6C-B1..B3 |
| `PT-BA` | Independent Campaign B auditor | W6C-C1..C3 |
| `PT-D13` | Campaign B digest manifest (docs) | W6C-E1..E3 |

**Not in this ticket:** `PT-C01` (Campaign C sample) → T-IGX-052-G; `PT-Q05`..`PT-Q12` → T-IGX-052-G; `PT-Q11` / M3 → T-IGX-052-H.

## 4. This batch (governance refinement only)

| Item | Path | Change |
|------|------|--------|
| Ticket | `governance/tickets/T-IGX-052-F.md` | refine scope, write set, test contract, lifecycle |
| Plan | this file | refine purpose, prerequisites, verification |

No edits under `docs/project/maintainers/qualification/`, `intergrax/`, `agents/`, `applications/`, `scripts/`, or `tests/` in this batch.

## 5. Future execution write set (out of scope here)

When plan-approved and program prerequisites are met, a separate implementation batch may touch only paths named in prep checklists and WP-12, for example:

- `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_b/*` (charter, manifest, results — W6C-E3)
- Evidence paths under `docs/project/maintainers/qualification/` per AR-08/AR-10/AR-53

That batch requires its own digest, approval, and `igx_gov` profile; it is **not** this refinement.

## 6. Verification plan (this batch)

- `uv run python -m scripts.governance.igx_gov ticket` on the worktree head (no pytest, no docs pair checker).
- `uv run python scripts/governance/plan_digest.py` on this plan; ticket `Plan digest:` updated to match.
- `uv run python -m scripts.governance.verify_ticket_lifecycle` on `T-IGX-052-F.md` if it runs without extra setup.

Reviewer check: `git diff --stat e69e1ff58...HEAD` lists exactly the two paths in section 4 for ticket F (this commit may include sibling tickets F/G/H in one governance batch per operator packet; if combined, diff must list only the six governance paths).

## 7. Out of scope

Campaign runs; auditor sessions; omega @ FCS claims; M3 pin; W7 lock; product paths; `scripts/` or `tests/` changes; FRZ promotion; editing `W6_CAMPAIGN_PREP_CHECKLIST.md` in this batch; merging or GitHub settings.

## 8. Risks

- Prep checklists remain Polish-labelled banners; tickets and plans stay English per VCS policy.
- Prerequisites may change when FCS or W5 finals land; digest must be recomputed after any plan edit.