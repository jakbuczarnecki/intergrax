# Delivery DevOS (DCP) — closure program backlog

**Authority:** complements [`GOVERNANCE-MANIFEST.yaml`](./GOVERNANCE-MANIFEST.yaml). Phases GOV-00..GOV-05 and GOV-07 are **implemented** at the procedural/technical baseline delivered in batch `T-IGX-001-A` (`2eddd4223`). This document is the **SSOT for remaining closure work** through external G6 handoff.

**Not in scope here:** FRZ-* promotion, ENT-AUDIT-X OD-ENT-* operator packets, product qualification — cross-linked only where DCP enforcement depends on GitHub admin actions.

## Gap register (from post phase 0–7 audit)

| Gap ID | Description | Wave | Ticket |
|--------|-------------|------|--------|
| GAP-GOV-06 | Optional `TRUSTED_POLICY_PIN.yaml` + CI pin compare (advisory) | W2 | `T-IGX-005-A` |
| GAP-POL-REG | `validate_policy_registry.py` + **governance-fast** CI tier wiring | W3 | `T-IGX-006-A` |
| GAP-OD-05 | Plan-verify / ticket FSM **server-side hardening** (beyond procedural `task_lifecycle.yaml`) | W4 | `T-IGX-007-A` |
| GAP-SYNC-AD | `igx_gov sync-adapters` packs for Claude/Codex (optional; Cursor matrix today) | W5 | `T-IGX-008-A` (partial) |
| GAP-ROLE-RT | Unit/integration **`test_agent_role_routing`** — subagent_role_map enforcement | W6 | `T-IGX-008-A` (partial) |
| GAP-BP-HOOK | **Branch protection** required checks on `development` + **hooks rollout** evidence | W7 | operator + `T-IGX-008-A` |
| GAP-G6-HO | **G6 / external audit** handoff; manifest `global_certification` flags | W8 | independent session |
| GAP-CERT | `certification_ready` / `dcp_program_certified` remain **false** until audit | W8 | manifest only |

Prior doc-only follow-ups (`T-IGX-003-A`) are **closed**; they do not replace execution waves below.

---

## Closure waves

### W1 — Closure program SSOT

| Field | Value |
|-------|-------|
| **Objective** | Publish this backlog, ticket numbering from batch **004+**, DevOS hub links, manifest `closure_backlog` pointer. |
| **Deliverables** | `governance/DCP-CLOSURE-BACKLOG.md`, `governance/tickets/README.md` update, stub tickets 004–008, skill + devos hub cross-links. |
| **Verification** | `uv run python -m scripts.governance.igx_gov doctor` |
| **Enforcement** | ADVISORY |
| **Depends on** | GOV-00..07 baseline |
| **Ticket** | `T-IGX-004-A` |

### W2 — GOV-06 trusted policy pin (optional)

| Field | Value |
|-------|-------|
| **Objective** | Implement optional policy pin artifact + compare job; keep **advisory** until operator promotes to SERVER. |
| **Deliverables** | `governance/TRUSTED_POLICY_PIN.yaml` (or documented deferral record), CI workflow job `governance-policy-pin`, manifest GOV-06 `status: implemented` only when job exists. |
| **Verification** | `uv run pytest tests/unit/governance/test_dcp_*.py -q`; pin job green on PR |
| **Enforcement** | ADVISORY → TECHNICAL (operator choice) |
| **Depends on** | W1 |
| **Ticket** | `T-IGX-005-A` |

**Operator note:** Do **not** implement in documentation-only batches; request explicitly before W2 code.

### W3 — Policy registry validation + governance-fast CI

| Field | Value |
|-------|-------|
| **Objective** | Fail closed when `policy-registry.yaml` references missing artifacts; expose **governance-fast** profile in CI tier map. |
| **Deliverables** | `scripts/governance/validate_policy_registry.py`, registry tests, `.github/workflows` governance-fast job or profile flag documented in [`CI-TIER-MAP.md`](../docs/project/technical/devos/CI-TIER-MAP.md). |
| **Verification** | `uv run python scripts/governance/validate_policy_registry.py`; `uv run pytest tests/unit/governance/ -q` (single invocation per verification step) |
| **Enforcement** | TECHNICAL |
| **Depends on** | W1 |
| **Ticket** | `T-IGX-006-A` |

### W4 — Plan-verify hardening (OD-05)

| Field | Value |
|-------|-------|
| **Objective** | Move plan approval guards from **TARGET DESIGN** to enforced transitions (ticket state + `verify_plan_approval.py` in CI/pre-push where appropriate). |
| **Deliverables** | Hardened `task_lifecycle.yaml` guards, optional `verify_ticket_lifecycle.py`, docs in [`INDEPENDENT-REVIEW-HANDOFF.md`](../docs/project/technical/devos/INDEPENDENT-REVIEW-HANDOFF.md) boundary; operator **OD-05** record when FSM is binding. |
| **Verification** | `uv run python scripts/governance/verify_plan_approval.py <ticket> --plan <plan>`; DCP unit tests |
| **Enforcement** | TECHNICAL → SERVER (with branch protection) |
| **Depends on** | W3 |
| **Ticket** | `T-IGX-007-A` |

### W5 — Sync-adapters (Claude / Codex) — optional

| Field | Value |
|-------|-------|
| **Objective** | Extend `igx_gov sync-adapters` beyond Cursor stub to regenerate adapter fragments for other harnesses without breaking GOV-02 Cursor path. |
| **Deliverables** | Adapter pack templates under `governance/agent-control-plane/adapters/`, CLI subcommands or documented manual sync, manifest GOV-02 note update. |
| **Verification** | `uv run python -m scripts.governance.igx_gov sync-adapters`; `verify_control_plane_consistency.py` |
| **Enforcement** | ADVISORY |
| **Depends on** | W1 |
| **Ticket** | `T-IGX-008-A` (optional slice) |

### W6 — Agent role routing tests

| Field | Value |
|-------|-------|
| **Objective** | Automated proof that `subagent_role_map` caps and role/skill bindings match registry SSOT. |
| **Deliverables** | `tests/unit/governance/test_agent_role_routing.py` (or extend `test_dcp_control_plane_registry.py`), registry invariant docs. |
| **Verification** | `uv run pytest tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_dcp_control_plane_registry.py -q` |
| **Enforcement** | TECHNICAL |
| **Depends on** | W1 |
| **Ticket** | `T-IGX-008-A` |

### W7 — Branch protection + hooks rollout

| Field | Value |
|-------|-------|
| **Objective** | Required status checks on `development` (push gate / governance tier); org-wide **git hooks** install evidence for Delivery DevOS contributors. |
| **Deliverables** | Operator evidence per ENT runbooks (`OD-ENT-02` pattern); DCP doc checkpoint in [`DEVELOPER-QUALITY-PROCESS.md`](../docs/project/technical/devos/DEVELOPER-QUALITY-PROCESS.md); no false claim of FRZ enforceability. |
| **Verification** | `gh api repos/:owner/:repo/branches/development/protection` (operator); `uv run python scripts/dev/verify_push_gate_readiness.py` |
| **Enforcement** | **SERVER** (GitHub admin + hooks local) |
| **Depends on** | W3, W4 (recommended) |
| **Ticket** | Operator-owned; track under `T-IGX-008-A` checkpoint |

### W8 — G6 external audit handoff + manifest certification

| Field | Value |
|-------|-------|
| **Objective** | Independent session per [`AUDIT_PROTOCOL.md`](../docs/audit_results/AUDIT_PROTOCOL.md); flip manifest flags only on auditor + operator signoff. |
| **Deliverables** | Audit artifact under `docs/audit_results/`; tickets at `ready_for_audit` → `closed`; **`global_certification.dcp_program_certified`** and **`g6_independent_audit_pass`** only after external PASS (never by implementer). |
| **Verification** | Separate chat with `@intergrax-independent-audit`; `uv run python -m scripts.governance.igx_gov doctor` on pinned SHA |
| **Enforcement** | PROCEDURAL + SERVER |
| **Depends on** | W2–W7 as applicable |
| **Ticket** | N/A (audit session); GOV-08 phase tracks closure |

---

## Agent execution protocol (normative for implementers)

### Session role selection

| Mode | When |
|------|------|
| **Orchestrator** | Closure wave spans multiple independent trees, CI investigation, or read-only fan-out; load `@intergrax-agent-roles` / default `orchestrator` from [`role_registry.yaml`](./agent-control-plane/role_registry.yaml). |
| **Single-role** | One ticket, one worktree, one cohesive diff (typical implementer / tester / delivery-closure). |

Orchestrator **must not** commit product paths; parent synthesizes subagent output, runs verification, commits with `Ref:` trailer.

### Parallelization rules

1. **Independent file trees** → parallel Task subagents (`explore`, `generalPurpose`, `ci-investigator`) with **max 2–3 concurrent** (registry caps: `explore` 3, `generalPurpose` 2).
2. **One pytest / one `uv run pytest`** per verification step — [`.cursor/rules/intergrax-test-process-budget.mdc`](../.cursor/rules/intergrax-test-process-budget.mdc).
3. **Isolated code attempts** → `best-of-n-runner` or separate git worktree ([`WORKTREES.md`](./workflows/WORKTREES.md)); `requires_worktree: true` in registry.
4. **Subagent_role_map** (authoritative):

   | Subagent | Maps to | Parallel max | Read-only |
   |----------|---------|--------------|-----------|
   | explore | architect-planner | 3 | yes |
   | generalPurpose | orchestrator | 2 | no |
   | ci-investigator | tester | 1 | yes |
   | bugbot | reviewer | 1 | yes |
   | security-review | independent-quality-auditor | 1 | yes |
   | best-of-n-runner | implementer | 2 | no (worktree) |

5. **Multitask / batch:** one ticket = one worktree. Split closure waves into **parallel tickets** only when **no shared file conflicts** (e.g. W2 pin file vs W6 tests-only).
6. **Pre-push:** `uv run python -m scripts.governance.igx_gov ticket`; document smoke profile for risky diffs in ticket **Test contract**.
7. **G6:** implementer **never** sets `ready_for_audit` to PASS or edits `global_certification.*` to true; use independent session + `AUDIT_PROTOCOL`.

### Suggested parallel fan-out by wave

| Waves | Parallel strategy |
|-------|-------------------|
| W2 + W6 | Pin CI vs routing tests — parallel if W2 skipped |
| W3 + W5 | Registry validator vs adapter docs — parallel read-only explore then serial merge |
| W7 | Operator GitHub + local hooks doc — **no agent** on branch protection API |

---

## Definition of Done — DCP program **CLOSED**

Distinct from product FRZ / ENT-AUDIT-X closure:

1. Waves **W1–W7** delivered or explicitly **waived in writing** by operator (W2/W5 optional waivers documented in manifest notes).
2. `uv run python -m scripts.governance.igx_gov doctor` green on `development` @ recorded SHA.
3. Governance unit suite (`tests/unit/governance/test_dcp_*.py` + W6 routing tests when present) green in CI **governance-fast** or enterprise tier.
4. Branch protection + required checks **live** on `development` (SERVER evidence) OR documented EXTERNAL_BLOCKED with operator ticket reference.
5. **W8:** independent G6 audit session completed; auditor attestation stored; operator sets manifest `GOV-08` / `global_certification` only per [`EVIDENCE-AND-CERTIFICATION.md`](../docs/project/technical/devos/EVIDENCE-AND-CERTIFICATION.md).
6. All closure tickets `T-IGX-004-A`..`008-A` (and any `-B` splits) **closed** with checkpoint paste.

Until (5), **`dcp_program_certified`** and phase **`certification_ready`** remain **false**.

---

## Rollout checklist (W7 — operator + docs)

| Step | Command / artifact | Enforcement |
|------|-------------------|-------------|
| Hooks | `uv run python -m scripts.governance.igx_gov bootstrap` | LOCAL |
| Push gate readiness | `uv run python scripts/dev/verify_push_gate_readiness.py` | LOCAL |
| Doctor | `uv run python -m scripts.governance.igx_gov doctor` | TECHNICAL |
| Branch protection | [`governance/rollout/BRANCH-PROTECTION-CHECKLIST.md`](./rollout/BRANCH-PROTECTION-CHECKLIST.md) + `scripts/dev/apply_github_branch_protection.ps1` | SERVER |
| CI governance-fast | `.github/workflows/unit-tests.yml` job `governance-fast` | TECHNICAL |
| GOV-06 pin job | same workflow job `governance-policy-pin` (`continue-on-error`) | ADVISORY |

If `gh api` branch protection cannot run (auth), record **EXTERNAL_BLOCKED** in ticket checkpoint — do not claim checks are required on GitHub.

---

## References

- [`governance/tickets/README.md`](./tickets/README.md)
- [`docs/project/technical/devos/README.md`](../docs/project/technical/devos/README.md)
- [`.cursor/skills/igx-devos-parallel-orchestration/SKILL.md`](../.cursor/skills/igx-devos-parallel-orchestration/SKILL.md)
- [`T-IGX-003-A`](./tickets/T-IGX-003-A.md) — prior gap doc closure
