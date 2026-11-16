# T-IGX-044-C — EQO-A remediation node PT-E02 (T-ENF-2): ticket-approval validators, legacy-exceptions file, ticket-governance job, Ref gate CI mode

## Scope

Finding EQOA-A6-002 (the ticket approval validators accept any line that starts with `Approved:`) and the ticket-level part of A2-038 (independence rests on string inequality). Adds, without changing the default of any existing validator: a strict approval grammar and a delegation-reference resolver (`scripts/governance/ticket_approval.py`, `scripts/governance/verify_ticket_approvals.py`), a closed legacy-exceptions file for the four unresolvable delegation references and the 19 files without an approval line (`governance/tickets/legacy_approval_exceptions.json`, existing tickets untouched, DP-68), a `--ci` mode of the commit Ref gate (`scripts/governance/verify_commit_ticket_refs.py`: skip environment ignored, unreadable range is an error, Ref must name a tracked ticket), a non-required workflow `ticket-governance` in a new file (`.github/workflows/ticket-governance.yml`) and the unit tests. Not part of this ticket and left for the Operator's personal ruling or for other nodes: making the job a required check (PT-S05), wiring the strict verifier into `igx_gov ticket`, flipping the default of `verify_plan_approval.py` / `verify_ticket_lifecycle.py`, the IIAC material files and the IIAC-side identity evidence (PT-E05), the replay over the 115 tickets (PT-V06). It is not a certification verdict.

## Plan digest

Plan digest: 8bde1efe9c2a84750c25dd63ce9fbdc8a9ef0575a30824441fc9ef8e70517d3e

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-044-C.md`](plans/PLAN_T-IGX-044-C.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line is written after plan review by the Operator or by a commissioned delegate session (in the delegate form with its `Delegation:` and `Delegate session:` lines); the implementer never writes it. This placeholder deliberately avoids the approval keyword.

## Implementer

Implementer session: local_a1d4a5fb-8ff6-4862-95fc-f40594e4386c

The independent G6 audit `AUD-ENF-1` (node `PT-AUD-ENF1`) of this ticket must be done by a separate session that is not the implementer.

## Test contract

One process at a time, under the shared pytest lock (`D:/Projekty/_wt/_locks/pytest.lock`, at most 15 minutes):

- `uv run pytest tests/unit/governance/test_ticket_approval_validator.py tests/unit/governance/test_ticket_ref_ci_mode.py tests/unit/governance/test_dcp_commit_ticket_refs.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_dcp_control_plane_registry.py -q -p no:cacheprovider`
- `uv run ruff check` and `uv run ruff format --check` on the changed Python files.
- `python -I scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-044-C.md` equals the digest above.
- The replay of the strict verifier over all tickets is node PT-V06, not run by this ticket.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the nine files of the plan's write set.

## Out of scope

`intergrax/`, `agents/`, `applications/`, the six `MATERIAL_GOVERNANCE_PATHS` of `scripts/governance/iiac_activation_gate.py`, the IIAC scripts and protocol, `iiac_g0_inventory.py`, existing tickets and plans, `governance/tickets/README.md`, `.github/workflows/unit-tests.yml` and `enterprise-audit*.yml`, GitHub settings, rulesets, required checks, auto-merge, `igx_gov`, `pyproject.toml`, `uv.lock`, FRZ promotion, the roadmap and the FRZ checklist.

## Lifecycle state

plan_pending
