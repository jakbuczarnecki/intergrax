# T-IGX-044-E — EQO-A remediation node PT-E04 (T-ENF-4): new IIAC corpus, artifact and delegation verifiers

## Scope

Adds NEW files only: three read-only verifiers (`verify_iiac_manifest_corpus.py`, `verify_iiac_manifest_artifacts.py`, `verify_iiac_delegation_refs.py`), one shared helper module, one allowlist of the three published invalid draft manifests, and a corpus of positive and negative example manifests, tickets and delegation references with expected verdicts, plus their tests. Source: EQO-A Campaign A findings A2-038 (part), A2-006, A6-002; WP-7 plan B design V1; decisions DP-58, DP-66, DP-68. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`.

Out of scope: any edit of an existing file under `scripts/` or `tests/`, the five IIAC scripts, the IIAC protocol, the six `MATERIAL_GOVERNANCE_PATHS`, `verify_plan_approval.py`, `verify_ticket_lifecycle.py`, every product path, `.github/`, wiring of the new verifiers into CI, hooks or any gate (that is PT-E05 / T-ENF-5 and needs SCOPE), the legacy-exceptions file (PT-E02), and any PASS, closure, FRZ promotion or certification claim.

## Plan digest

Plan digest: c95da68be4359ceec0466a305f28782913054d694a704c27a0c6078e25b23f56

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-044-E.md`](plans/PLAN_T-IGX-044-E.md). The digest is computed with `scripts/governance/plan_digest.py` and must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line is written after plan review by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time under the shared pytest lock, at most 15 minutes:

- `uv run pytest tests/unit/governance/test_iiac_new_verifiers.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_g0_inventory.py -o cache_dir=<scratch dir>`
- `ruff check` and `ruff format --check` over the new Python files.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the 19 paths of the plan's write set, all added.
- Independent audit AUD-ENF-2 at the exact head SHA by a session that did not implement or review it.

## Out of scope

See Scope. No Docker, vendor integration or production configuration.

## Lifecycle state

plan_approved
