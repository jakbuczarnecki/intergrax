# AGENT-OPS-X v2.1 — transport plan (final, technical corrections applied)

> **How this file is used.**
>
> This file is a transport container. W0-P01 commits it verbatim as `governance/campaigns/AGENT-OPS-X/source/plan-v2.1.md` and splits it at the file markers into the campaign tree. A file marker is an HTML comment of the form `igx:file` followed by a path.
>
> Each section follows the canonical format defined in `contracts/authorization.md`:
> - one H1 line
> - an `igx:immutable` block
> - optionally, mutable blocks
>
> Each worker loads only its own packet and the contracts it names. Machine-readable packet contracts live in `contracts/packets.md`.
>
> The Planning annex at the end is for Operator review only and is not split.

<!-- igx:file governance/campaigns/AGENT-OPS-X/charter.md -->
# AGENT-OPS-X — campaign charter

<!-- igx:immutable begin -->
## 1. Purpose and principles

Build a measurable agent working model for InterGRAX that is enforced mechanically, not by instruction.

- **Deterministic first.** No LLM is used for work that code can do reliably: status, tests, CI, scheduling, publication, resume.
- **Minimum viable context.** A worker receives its packet, the read ranges and a failure digest, nothing more.
- **Autonomous by default.** Approval authorizes full execution within §5–§7. Execution stops only on E1–E8 (`contracts/escalation.md`).
- **Simplest sufficient mechanism.** Use `igx_gov` and the DCP, native Claude Code, herdr `hc`, git, GitHub rulesets, and stdlib SQLite. Add no new infrastructure, no MCP and no paid service.

Quality, security, IIAC, G6 and Enterprise standards are never lowered.

## 2. Baseline

Recorded in `evidence/baseline.md`, with the figures in Planning annex H.
- 5.62B tokens; estimated $2,961 at list price.
- 58 % of cost comes from subagents.
- Main-session context averages 357k tokens.
- Effort is always `high`.
- 80 % of Operator messages are go-aheads.
- 493 wake-up turns.

## 3. Scope

The authoritative, machine-readable scope is `contracts/packets.md`:
- `charter.edit_globs`: what may be edited
- `charter.forbidden_globs`: what may never be edited (E7)
- `charter.operator_applied`: files the campaign prepares as patches and only the Operator applies

`boot.py validate` and `gate.py` enforce these lists. Summary:
- Edits are limited to the agent environment, governance tooling, campaign files and tests.
- Product tiers, the IIAC material set, frozen-campaign files and `agent-env/plans/**` are forbidden.
- Workflows, CODEOWNERS and rulesets are applied by the Operator only.

## 4. Pre-activation acceptance (all verifiable before activation; method in `contracts/measurement.md`)

| ID | Criterion | Target |
|---|---|---|
| AC-01a | Total processed tokens per comparable task (input + cache-write + cache-read + output) | ≥ 60 % reduction, stretch 80 % |
| AC-01b | Estimated API cost per comparable task | ≥ 60 % reduction |
| AC-01c | Claude Max limit consumption per task, from `rate_limits` where telemetry exists | reported; target ≥ 60 % where measurable |
| AC-02 | Wall time per comparable task | ≥ 50 % reduction |
| AC-03 | Trivial continuation questions in W4 runs | 0 |
| AC-04 | LLM packets executed **by the running engine** without unplanned Operator intervention (n ≥ 20; bootstrap packets do not count) | ≥ 95 % |
| AC-05 | Main-session first-turn context | ≤ 65k, or an evidence-backed floor |
| AC-06 | Median tokens per comparable worker run | ≤ 2M |
| AC-07 | Opus share of routine worker runs | ≤ 10 % |
| AC-08 | Unauthorized operations, gate bypasses, Enterprise downgrades, recovery state loss or double publication, agent access to Operator or publisher credentials | 0 |
| AC-09 | Total campaign spend | ≤ $800 est. |
| AC-10 | Independent G6 audit | PASS |

**Rules for evaluating the criteria:**
- A model swap is never counted as a token reduction.
- If a target is missed for a reason outside the project's control, report the evidence, the value that was achieved, and the alternatives.
- Criteria are never lowered silently.

**Post-activation health check.** This is not a closure condition. The first roadmap wave after activation re-measures AC-03, AC-04 and the budget KPIs. On failure the engine pauses new waves and raises E4.

## 5. Identity, isolation, authority

**Three Windows accounts:**

| Account | Runs | Holds credentials |
|---|---|---|
| Operator | the Operator's own work | Operator credentials |
| `igx-agent` (non-admin) | every LLM process and the engine: workers, bootstrap and console sessions | none |
| `igx-pub` (non-admin) | only `ao/publisher.py` | the GitHub App private key |

**Rules for `igx-pub` and the publisher:**
- `igx-agent` cannot read the profiles of the Operator or `igx-pub`, and has no rights on `D:\Projekty\intergrax` or the publisher's trusted clone.
- The publisher runs as a Scheduled Task under `igx-pub`.
- It executes only code from its trusted clone at `development`, which contains only Operator-approved commits. The single bootstrap exception is OA-2a.
- It never executes code or git hooks inside the agent clone. It fetches `agent-ops*` refs from the agent clone into its own clone, then pushes from there.
- It accepts only typed requests from the spool (`contracts/publication.md`).

**GitHub App.**
- Owned by the Operator's personal account and installed on this repository only.
- Permissions: Contents RW, Pull requests RW, Metadata R.
- No Workflows, Administration, Commit statuses, Checks, Actions or Secrets permissions.
- Never a ruleset bypass actor.
- Tokens last 1 hour and are minted only inside the publisher.

**Fallback, only if the W0 publisher selftest fails on the App path:** a machine user with a classic `repo` PAT without the `workflow` scope, Write role, also used only through the publisher. Statuses it could post are ineffective, because the required checks are pinned to the GitHub Actions source. Fine-grained PATs are not supported for collaborators on another user's repository, so they are not an option.

**Server-side enforcement.** The ruleset order is in `contracts/publication.md`. The final `development` ruleset:
- requires a PR
- dismisses stale approvals
- requires code-owner review (CODEOWNERS read from the base branch), with 0 additional required approvals
- requires the checks `agent-ops-gate`, `agent-ops-tests` and `ci-smoke`, pinned to the GitHub Actions source, with the strict policy
- blocks force-push and deletion
- has no bypass actors

**Authority.**
- The Operator's GitHub `APPROVED` review at the exact commit. For the bootstrap form, see §5a.
- Claude Code permissions are accident prevention, not authorization.

**Subscription only.**
- Workers run on `igx-agent`'s Claude Max login.
- No `--bare`, no API or provider variables, no `apiKeyHelper`.
- The engine checks the auth source of every init record.
- On a rate limit, wait for the reset. Never fall back to paid usage.
- Extra usage stays disabled.

**Residual risk.** This is the real boundary, stated honestly: `contracts/publication.md` §Residual.

## 5a. Bootstrap authority

1. Plan approval of this transport plan is recorded in tickets A–F (legacy form, listed in `legacy_tickets.txt`).
2. The bootstrap ruleset requires Operator code-owner approval of every PR.
3. The Operator's approval of the W0 PR at its head is the charter approval.
4. W2-P08 re-binds the tickets to it.

## 6. Budget

- **Cap.** $800, estimated at list price. This is an estimate equivalent, not an authorization to spend.
- **Covers** all sessions, retries, workers and benchmarks.
- **Allocations:**

  | W0 | W1 | W2 | W3 | W4 | W5 |
  |---|---|---|---|---|---|
  | $60 | $90 | $170 | $260 | $150 | $70 |

- **Moving money between waves:**
  - unused allocation rolls forward
  - borrowing from later waves is allowed within the total
  - a $60 reserve for W5 cannot be borrowed
- **Atomic reservation** happens before every launch (`contracts/budget.md`).
- This is local enforcement, partly delayed. It is not a server-side guarantee.

## 7. Standing pre-authorizations

| ID | Authorization |
|---|---|
| A-01 | Edit within `charter.edit_globs` through packets. Worktrees go in `_agent/wt/*`; branches are `agent-ops/w<N>`, `agent-ops/w<N>-<packet>`, `agent-ops-bench/*` and `agent-ops/probe-*`. |
| A-02 | The publisher pushes those branches and creates, updates, marks ready and comments on PRs to `development`, at the request of `boot.py` or the engine. |
| A-03 | Merges: M0/M1 after the server-side gates; M2/M2S after the Operator's code-owner approval at the head SHA. M3 is never merged by agents. |
| A-04 | One TRUSTED_POLICY_PIN re-pin per wave PR. |
| A-05 | `claude -p` workers run as `igx-agent` within budget, subscription only. |
| A-06 | Changes to `igx-agent`'s user settings and hc config. Changes to the Operator's profile are delivered as an optional patch only. |
| A-07 | Probes and the publisher selftest: haiku, 1 turn, a few cents; scratch branches `agent-ops/probe-*`, deleted afterwards. |
| A-08 | W4 benchmarks and pilots only on `agent-ops-bench/*`. Their draft PRs are closed without merging. |
| A-09 | W5 independent G6 auditor sessions, each separate. This is the Operator authorization the G6 rule requires. |
| A-10 | Bounded corrective packets (`contracts/routing.md`), including those from W5 findings, within the original goal, scope, budget and quality. New decisions or scope go to E1/E3. |
| A-11 | Migration drafts for frozen campaigns, without touching their files. The migrations themselves happen only after activation. |

## 8. Operator actions

The complete list is `contracts/operator-actions.md`.
<!-- igx:immutable end -->

<!-- igx:mutable begin -->
Live state: `boot.py status` until W3-P10, then `igx_gov wave status AGENT-OPS-X`.
<!-- igx:mutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/decisions.md -->
# AGENT-OPS-X — decision ledger

<!-- igx:immutable begin -->
| ID | Decision (Operator, 2026-11-20) | Binding conditions |
|---|---|---|
| OD-AX-01 | The approval unit is a campaign or wave charter. A delegate is used only for G6. | Fail-closed immutable digest |
| OD-AX-02 | Stop guard driven by state, plus a rule; regex is used only for diagnostics | `contracts/stop-guard.md` |
| OD-AX-03 | Native capabilities first; ccusage only as an optional offline tool; no MCP | — |
| OD-AX-04 | Publication classes M0–M3 and M2S | `contracts/publication.md` |
| D-01 | Separate agent identity; Operator approvals are GitHub reviews | Real isolation: a separate Windows user for LLM processes and a separate one for the publisher that holds the key; a GitHub App with fallback; no bypass |
| D-02 | Server-side enforcement through rulesets, CODEOWNERS and Actions-pinned checks | Mandatory under D-01. A recorded break-glass procedure (OA-7) covers ruleset deadlock |
| D-03 | `hc` + `claude -p`, with a subprocess fallback | Max subscription only; CLI flags verified locally |
| D-04 | $800 est. cap | Reallocation; W5 reserve; atomic reservations |
| D-05 | Every unfinished campaign stays frozen until activation | W0-P07 snapshot; only the listed W4 pilots, on bench branches, never merged |
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/waves.md -->
# AGENT-OPS-X — wave index (human view of `contracts/packets.md`)

<!-- igx:immutable begin -->
The DAG, the scopes, the test commands and the executors live **only** in the JSON in `contracts/packets.md`. `boot.py validate` fails if this file disagrees with it.

**Waves:**

| Wave | Scope |
|---|---|
| W0 | setup, tools, probes, baseline |
| W1 | token reduction |
| W2 | authorization and gates |
| W3 | deterministic engine |
| W4 | qualification |
| W5 | audit and activation |

**Branching and PR model:**
- Each packet is developed on `agent-ops/w<N>-<packet>`, in its own worktree cut from the tip of the wave branch `agent-ops/w<N>`.
- Integration is a `--no-ff` merge carrying the trailer `Packet: <id>`. After integration the packet is `implemented`.
- A dependency inside a wave needs `implemented`. A dependency across waves also needs `implemented` (the next wave branch starts from the previous wave's tip). Steps that need a merge state say so explicitly through `external` preconditions.
- There is one PR per wave. It opens when every packet is `implemented` and the previous wave's PR has merged. The publisher first merges `development` into the wave branch.
- A merged wave PR makes its packets `published`. Wave PRs are sequential, so `development` does not move while an approval is pending.
- An external precondition such as an Operator approval or OA-3 sets `waiting_external`. Other ready packets continue. Nothing is asked.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/packets.md -->
# Contract — machine-readable packet contracts (authoritative DAG)

<!-- igx:immutable begin -->
Parsed by `boot.py`, `scheduler.py` and `gate.py` with `json.loads` of the single fenced block below. Field semantics:

| Field | Meaning |
|---|---|
| `deps` | Packet ids that must be `implemented`. |
| `external` | Named events that `boot.py` or the engine detects deterministically: `w0_pr_approved`, `w2_pr_merged`, `oa2a_publisher_running`, `oa3_applied`, `pilot_charter_approved`, `w5_pr_merged`. |
| `provides` / `uses` | Tool ids. `uses` must be in `preexisting` or provided by a transitive dep. |
| `read_scope` / `edit_scope` | Globs. `edit_scope` must be covered by `charter.edit_globs` and must not intersect `forbidden_globs`. |
| `test_commands` | Each command must match the executor's allow rules. |
| `ac04` | The number of LLM packets the run contributes to AC-04. |

```json
{
 "charter": {
  "edit_globs": ["agent-env/**", ".claude/**", ".cursor/**", "CLAUDE.md", "AGENTS.md",
   "governance/campaigns/AGENT-OPS-X/**", "governance/agent-control-plane/**", "governance/tickets/README.md", "governance/tickets/T-IGX-*.md",
   "governance/TRUSTED_POLICY_PIN.yaml", "governance/dcp-commit-ref-config.yaml",
   "scripts/governance/agent_ops/**", "scripts/governance/igx_gov/**", "scripts/governance/verify_plan_approval.py",
   "scripts/governance/verify_ticket_lifecycle.py", "scripts/governance/sync_agent_env.py", "scripts/governance/ticket_lock.py",
   "scripts/hooks/**", "scripts/ci/run_push_gate.py", "tests/unit/governance/**",
   "docs/project/technical/devos/**", "docs/project/technical/guides/AGENT_TOKEN_ECONOMY.md", "docs/project/technical/guides/AGENT_INSTRUCTIONS.md",
   "docs/project/maintainers/bootstrap/hep_step.txt", "docs/project/maintainers/plans/ROADMAP_INDEX.md", "pyproject.toml", "uv.lock"],
  "conditional": {"pyproject.toml": "only if W0-P02 reports pyjwt/cryptography not importable", "uv.lock": "same"},
  "forbidden_globs": ["intergrax/**", "agents/**", "applications/**", "agent-env/plans/**", ".github/**",
   "scripts/governance/verify_iiac_policy_activation.py", "scripts/governance/iiac_activation_gate.py", "scripts/governance/iiac_common.py",
   "scripts/governance/verify_iiac_audit_manifest.py", "scripts/governance/verify_iiac_role_separation.py",
   "**/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"],
  "frozen_campaign_globs_source": "governance/campaigns/AGENT-OPS-X/evidence/frozen-campaigns.md#globs",
  "operator_applied": [".github/workflows/agent-ops.yml", ".github/CODEOWNERS", "rulesets", "repository settings", "App settings"]
 },
 "preexisting": ["uv", "git", "rg", "hc", "claude", "scripts/governance/plan_digest.py", "scripts/governance/sync_agent_env.py",
  "igx_gov.ticket", "igx_gov.doctor", "governance/tickets/README.md#template"],
 "packets": [
  {"id":"W0-P00","file":"packets/W0-P00.md","class":"Op","pub":"-","executor":"operator","deps":[],"external":[],
   "provides":["os.igx-agent","os.igx-pub","app","agent-clone","trusted-clone","spool","bootstrap-settings","bootstrap-ruleset","inbox-plan"],
   "uses":[],"read_scope":[],"edit_scope":[],"test_commands":[],"ac04":0},
  {"id":"W0-P01","file":"packets/W0-P01.md","class":"S","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W0-P00"],"external":[],
   "provides":["ao.split_plan","ao.boot","ao.gh_read","ao.publisher","tickets","campaign-tree"],
   "uses":["inbox-plan","scripts/governance/plan_digest.py","governance/tickets/README.md#template"],
   "read_scope":["governance/tickets/README.md","scripts/governance/plan_digest.py","scripts/governance/igx_gov/cli.py"],
   "edit_scope":["governance/tickets/T-IGX-*.md","governance/campaigns/AGENT-OPS-X/**","scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops -q"],"ac04":0},
  {"id":"W0-P02","file":"packets/W0-P02.md","class":"S","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W0-P01"],"external":[],
   "provides":["ao.probe.isolation","ao.probe.capabilities","ao.probe.overhead","ao.probe.workflow_preflight","ao.probe.freeze_snapshot"],
   "uses":["ao.boot"],"read_scope":["scripts/governance/agent_ops/**",".github/workflows/**"],
   "edit_scope":["scripts/governance/agent_ops/probe.py","tests/unit/governance/agent_ops/**","governance/campaigns/AGENT-OPS-X/evidence/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_probe.py -q","uv run python -m scripts.governance.agent_ops.probe isolation","uv run python -m scripts.governance.agent_ops.probe workflow-preflight"],"ac04":0},
  {"id":"W0-P03","file":"packets/W0-P03.md","class":"D","pub":"M0","executor":"bootstrap","deps":["W0-P01"],"external":[],
   "provides":["evidence.baseline"],"uses":["ao.boot"],"read_scope":["governance/campaigns/AGENT-OPS-X/source/plan-v2.1.md"],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.agent_ops.boot run W0-P03"],"ac04":0},
  {"id":"W0-P04","file":"packets/W0-P04.md","class":"St","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W0-P01"],"external":[],
   "provides":["ao.usage","igx_gov.usage","ao.boot.budget"],"uses":["ao.boot"],
   "read_scope":["scripts/governance/igx_gov/cli.py","scripts/governance/agent_ops/boot.py"],
   "edit_scope":["scripts/governance/agent_ops/**","scripts/governance/igx_gov/cli.py","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_usage.py tests/unit/governance/agent_ops/test_boot.py -q"],"ac04":0},
  {"id":"W0-P05","file":"packets/W0-P05.md","class":"St","pub":"M0","executor":"bootstrap","model":"sonnet/medium","deps":["W0-P02"],"external":[],
   "provides":["evidence.enterprise_impact","evidence.act_n11_baseline","ao.probe.act_n11_baseline"],"uses":["ao.probe.workflow_preflight"],
   "read_scope":["tests/qualification/ebh_7/_constants.py","governance/TRUSTED_POLICY_PIN.yaml","docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md",".github/workflows/**","scripts/governance/iiac_activation_gate.py"],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**","scripts/governance/agent_ops/probe.py","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run python -m scripts.governance.agent_ops.probe act-n11-baseline"],"ac04":0},
  {"id":"W0-P07","file":"packets/W0-P07.md","class":"D","pub":"M0","executor":"bootstrap","deps":["W0-P02"],"external":[],
   "provides":["evidence.frozen_campaigns"],"uses":["ao.probe.freeze_snapshot","ao.gh_read"],"read_scope":["governance/tickets/*.md"],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.agent_ops.probe freeze-snapshot"],"ac04":0},
  {"id":"W0-P06","file":"packets/W0-P06.md","class":"S","pub":"M0","executor":"bootstrap","model":"opus/medium","deps":["W0-P04","W0-P07"],"external":[],
   "provides":["contracts.benchmark","ao.probe.bench_check"],"uses":["ao.usage","evidence.frozen_campaigns"],
   "read_scope":["governance/tickets/*.md","docs/project/maintainers/plans/platform_enterprise_completion_roadmap.md"],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/contracts/benchmark.md","governance/campaigns/AGENT-OPS-X/evidence/**","scripts/governance/agent_ops/probe.py","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run python -m scripts.governance.agent_ops.probe bench-check"],"ac04":0},
  {"id":"W1-P01","file":"packets/W1-P01.md","class":"St","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W0-P03","W0-P05","W0-P06","W0-P07"],"external":[],
   "provides":["rule.core","gate.size"],"uses":["scripts/governance/sync_agent_env.py"],
   "read_scope":["CLAUDE.md","AGENTS.md","agent-env/rules/**","agent-env/README.md","scripts/governance/sync_agent_env.py"],
   "edit_scope":["agent-env/rules/**","agent-env/skills/**","agent-env/README.md","agent-env/size_exemptions.yaml","CLAUDE.md","AGENTS.md",".cursor/**",".claude/skills/**","scripts/governance/sync_agent_env.py","tests/unit/governance/test_agent_env_sync.py"],
   "test_commands":["uv run pytest tests/unit/governance/test_agent_env_sync.py -q","uv run python scripts/governance/sync_agent_env.py --check"],"ac04":0},
  {"id":"W1-P02","file":"packets/W1-P02.md","class":"S","pub":"M2","executor":"bootstrap","model":"sonnet/medium","deps":["W1-P01"],"external":[],
   "provides":["settings.skill_overrides"],"uses":["ao.probe.overhead"],"read_scope":[".claude/settings.json"],
   "edit_scope":[".claude/settings.json","docs/project/technical/guides/AGENT_TOKEN_ECONOMY.md"],
   "test_commands":["uv run python -m scripts.governance.agent_ops.probe overhead"],"ac04":0},
  {"id":"W1-P03","file":"packets/W1-P03.md","class":"St","pub":"M2","executor":"bootstrap","model":"sonnet/medium","deps":["W1-P02"],"external":[],
   "provides":["agents.igx"],"uses":["rule.core"],
   "read_scope":["governance/agent-control-plane/role_registry.yaml","agent-env/skills/igx-devos-*/**"],
   "edit_scope":["agent-env/agents/**",".claude/agents/**",".claude/settings.json",".claude/skills/**","agent-env/skills/**","governance/agent-control-plane/role_registry.yaml","governance/TRUSTED_POLICY_PIN.yaml","scripts/governance/sync_agent_env.py","tests/unit/governance/**"],
   "test_commands":["uv run pytest tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_agent_role_routing.py -q","uv run python scripts/governance/sync_agent_env.py --check"],"ac04":0},
  {"id":"W1-P04","file":"packets/W1-P04.md","class":"S","pub":"M2","executor":"bootstrap","model":"sonnet/medium","deps":["W1-P03"],"external":[],
   "provides":["settings.model_effort"],"uses":["ao.probe.capabilities"],"read_scope":[".claude/settings.json"],
   "edit_scope":[".claude/settings.json"],"test_commands":["uv run python -m scripts.governance.agent_ops.probe overhead"],"ac04":0},
  {"id":"W1-P05","file":"packets/W1-P05.md","class":"S","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W1-P01"],"external":[],
   "provides":["rule.no_polling"],"uses":["rule.core"],"read_scope":["agent-env/rules/intergrax-core.md"],
   "edit_scope":["agent-env/rules/intergrax-core.md"],"test_commands":["uv run python scripts/governance/sync_agent_env.py --check"],"ac04":0},
  {"id":"W1-P06","file":"packets/W1-P06.md","class":"S","pub":"M0","executor":"bootstrap","model":"sonnet/medium","deps":["W1-P01"],"external":[],
   "provides":["evidence.operator_profile_patch"],"uses":[],"read_scope":["governance/campaigns/AGENT-OPS-X/source/plan-v2.1.md"],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":[],"ac04":0},
  {"id":"W1-P07","file":"packets/W1-P07.md","class":"St","pub":"M2","executor":"bootstrap","model":"sonnet/medium","deps":["W1-P04"],"external":[],
   "provides":["hooks.read_guard","hooks.bash_guard"],"uses":["rule.core"],"read_scope":[".claude/settings.json"],
   "edit_scope":["scripts/hooks/**",".claude/settings.json","tests/unit/governance/**"],
   "test_commands":["uv run pytest tests/unit/governance/test_read_guard.py tests/unit/governance/test_bash_guard.py -q"],"ac04":0},
  {"id":"W1-P08","file":"packets/W1-P08.md","class":"D","pub":"M0","executor":"bootstrap","deps":["W1-P05","W1-P06","W1-P07"],"external":[],
   "provides":["evidence.w1_checkpoint"],"uses":["ao.probe.overhead","igx_gov.usage"],"read_scope":[],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.agent_ops.boot run W1-P08"],"ac04":0},
  {"id":"W2-P01","file":"packets/W2-P01.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W1-P08"],"external":[],
   "provides":["ao.charter","igx_gov.charter"],"uses":["campaign-tree"],"read_scope":["governance/campaigns/AGENT-OPS-X/contracts/authorization.md","scripts/governance/igx_gov/cli.py"],
   "edit_scope":["scripts/governance/agent_ops/**","scripts/governance/igx_gov/**","governance/agent-control-plane/agent_ops.json","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_charter.py -q"],"ac04":0},
  {"id":"W2-P02","file":"packets/W2-P02.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W2-P01"],"external":[],
   "provides":["ao.classify","ao.gate","patch.workflow"],"uses":["ao.charter","evidence.act_n11_baseline"],
   "read_scope":["governance/campaigns/AGENT-OPS-X/contracts/publication.md"],
   "edit_scope":["scripts/governance/agent_ops/**","governance/agent-control-plane/risk_map.json","governance/campaigns/AGENT-OPS-X/evidence/patches/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_classify.py tests/unit/governance/agent_ops/test_gate.py -q"],"ac04":0},
  {"id":"W2-P03","file":"packets/W2-P03.md","class":"St","pub":"M2S","executor":"bootstrap","model":"sonnet/medium","deps":["W2-P02"],"external":[],
   "provides":["ao.merge"],"uses":["ao.gate","ao.publisher"],"read_scope":["scripts/governance/agent_ops/publisher.py"],
   "edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_merge.py tests/unit/governance/agent_ops/test_publisher.py -q"],"ac04":0},
  {"id":"W2-P04","file":"packets/W2-P04.md","class":"St","pub":"M2S","executor":"bootstrap","model":"sonnet/medium","deps":["W2-P03","W1-P07"],"external":[],
   "provides":["settings.final","ao.worker_settings"],"uses":["hooks.read_guard"],"read_scope":[".claude/settings.json"],
   "edit_scope":[".claude/settings.json","scripts/governance/agent_ops/worker_settings/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_settings.py -q"],"ac04":0},
  {"id":"W2-P05","file":"packets/W2-P05.md","class":"St","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W2-P04"],"external":[],
   "provides":["igx_gov.lint_autonomy"],"uses":["rule.core"],
   "read_scope":["agent-env/rules/**","agent-env/skills/**","governance/agent-control-plane/MATRIX.md","governance/tickets/README.md","docs/project/technical/devos/**","docs/project/technical/guides/AGENT_INSTRUCTIONS.md","docs/project/maintainers/bootstrap/hep_step.txt"],
   "edit_scope":["agent-env/rules/**","agent-env/skills/**","agent-env/agents/**","agent-env/README.md","agent-env/autonomy_allowlist.yaml",".claude/skills/**",".cursor/**","governance/agent-control-plane/MATRIX.md","governance/tickets/README.md","docs/project/technical/devos/**","docs/project/technical/guides/AGENT_INSTRUCTIONS.md","docs/project/maintainers/bootstrap/hep_step.txt","scripts/governance/igx_gov/**","scripts/governance/agent_ops/**","tests/unit/governance/**"],
   "test_commands":["uv run python -m scripts.governance.igx_gov lint autonomy","uv run python scripts/governance/sync_agent_env.py --check"],"ac04":0},
  {"id":"W2-P06","file":"packets/W2-P06.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W2-P05"],"external":[],
   "provides":["lifecycle.unified"],"uses":["ao.charter","ao.gate"],
   "read_scope":["scripts/governance/verify_plan_approval.py","scripts/governance/verify_ticket_lifecycle.py","scripts/ci/run_push_gate.py","governance/agent-control-plane/*.yaml"],
   "edit_scope":["scripts/governance/verify_plan_approval.py","scripts/governance/verify_ticket_lifecycle.py","scripts/governance/ticket_lock.py","scripts/ci/run_push_gate.py","governance/agent-control-plane/**","governance/dcp-commit-ref-config.yaml","governance/TRUSTED_POLICY_PIN.yaml","governance/tickets/README.md","scripts/governance/agent_ops/**","tests/unit/governance/**"],
   "test_commands":["uv run pytest tests/unit/governance/test_dcp_control_plane_registry.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/agent_ops -q"],"ac04":0},
  {"id":"W2-P07","file":"packets/W2-P07.md","class":"S","pub":"M2S","executor":"bootstrap","model":"sonnet/medium","deps":["W2-P06"],"external":[],
   "provides":["patch.codeowners","spec.rulesets","ao.probe.github"],"uses":["ao.classify"],"read_scope":["governance/agent-control-plane/risk_map.json"],
   "edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/patches/**","governance/agent-control-plane/github/**","docs/project/technical/devos/**","scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_codeowners_sync.py -q"],"ac04":0},
  {"id":"W2-P08","file":"packets/W2-P08.md","class":"D","pub":"M2","executor":"bootstrap","deps":["W2-P06"],"external":["w0_pr_approved"],
   "provides":["approval.bound"],"uses":["ao.charter","ao.gh_read"],"read_scope":[],
   "edit_scope":["governance/tickets/T-IGX-*.md","governance/agent-control-plane/legacy_tickets.txt"],
   "test_commands":["uv run python -m scripts.governance.igx_gov charter verify AGENT-OPS-X"],"ac04":0},
  {"id":"W3-P01","file":"packets/W3-P01.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W2-P07"],"external":[],
   "provides":["ao.state"],"uses":["ao.boot"],"read_scope":["governance/campaigns/AGENT-OPS-X/contracts/state.md"],
   "edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],"test_commands":["uv run pytest tests/unit/governance/agent_ops/test_state.py -q"],"ac04":0},
  {"id":"W3-P02","file":"packets/W3-P02.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W3-P01"],"external":[],
   "provides":["ao.scheduler"],"uses":["ao.state"],"read_scope":["governance/campaigns/AGENT-OPS-X/contracts/packets.md"],
   "edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],"test_commands":["uv run pytest tests/unit/governance/agent_ops/test_scheduler.py -q"],"ac04":0},
  {"id":"W3-P03","file":"packets/W3-P03.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W3-P01"],"external":[],
   "provides":["ao.runner"],"uses":["ao.state","ao.probe.capabilities"],"read_scope":["governance/campaigns/AGENT-OPS-X/evidence/native-capabilities.md"],
   "edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],"test_commands":["uv run pytest tests/unit/governance/agent_ops/test_runner.py -q"],"ac04":0},
  {"id":"W3-P04","file":"packets/W3-P04.md","class":"St","pub":"M2S","executor":"bootstrap","model":"sonnet/medium","deps":["W3-P01"],"external":[],
   "provides":["ao.quality"],"uses":["ao.state"],"read_scope":[],"edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_quality.py -q"],"ac04":0},
  {"id":"W3-P05","file":"packets/W3-P05.md","class":"St","pub":"M2S","executor":"bootstrap","model":"sonnet/medium","deps":["W3-P03"],"external":[],
   "provides":["ao.review"],"uses":["ao.runner"],"read_scope":[],"edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_review.py -q"],"ac04":0},
  {"id":"W3-P06","file":"packets/W3-P06.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W3-P04","W2-P03"],"external":[],
   "provides":["ao.publish","ao.roadmap_index"],"uses":["ao.merge","ao.publisher"],"read_scope":["docs/project/maintainers/plans/platform_enterprise_completion_roadmap.md"],
   "edit_scope":["scripts/governance/agent_ops/**","docs/project/maintainers/plans/ROADMAP_INDEX.md","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_publish.py -q"],"ac04":0},
  {"id":"W3-P07","file":"packets/W3-P07.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W3-P03"],"external":[],
   "provides":["ao.budget"],"uses":["ao.state","ao.usage"],"read_scope":["governance/campaigns/AGENT-OPS-X/contracts/budget.md"],
   "edit_scope":["scripts/governance/agent_ops/**","tests/unit/governance/agent_ops/**"],"test_commands":["uv run pytest tests/unit/governance/agent_ops/test_budget.py -q"],"ac04":0},
  {"id":"W3-P08","file":"packets/W3-P08.md","class":"St","pub":"M2S","executor":"bootstrap","model":"sonnet/medium","deps":["W3-P01"],"external":[],
   "provides":["hooks.stop_guard","hooks.ctx"],"uses":["ao.state"],"read_scope":[".claude/settings.json"],
   "edit_scope":["scripts/hooks/**",".claude/settings.json","tests/unit/governance/**"],"test_commands":["uv run pytest tests/unit/governance/test_stop_guard.py -q"],"ac04":0},
  {"id":"W3-P09","file":"packets/W3-P09.md","class":"C","pub":"M2S","executor":"bootstrap","model":"sonnet/high","deps":["W3-P02","W3-P05","W3-P06","W3-P07","W3-P08"],"external":[],
   "provides":["igx_gov.wave"],"uses":["ao.scheduler","ao.runner","ao.quality","ao.review","ao.publish","ao.budget","hooks.stop_guard"],"read_scope":[],
   "edit_scope":["scripts/governance/agent_ops/**","scripts/governance/igx_gov/**","tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_resume.py -q"],"ac04":0},
  {"id":"W3-P10","file":"packets/W3-P10.md","class":"St","pub":"M1","executor":"bootstrap","model":"sonnet/medium","deps":["W3-P09"],"external":[],
   "provides":["engine.running"],"uses":["igx_gov.wave","hooks.stop_guard"],"read_scope":[],"edit_scope":["tests/unit/governance/agent_ops/**"],
   "test_commands":["uv run pytest tests/unit/governance/agent_ops/test_fault_injection.py -q"],"ac04":0},
  {"id":"W4-P01","file":"packets/W4.md","class":"D","pub":"M0","executor":"engine","deps":["W3-P10"],"external":[],"provides":["evidence.benchmark"],"uses":["engine.running","contracts.benchmark"],
   "read_scope":["governance/campaigns/AGENT-OPS-X/contracts/benchmark.md"],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":[],"ac04":8},
  {"id":"W4-P02","file":"packets/W4.md","class":"EC","pub":"bench","executor":"engine","deps":["W3-P10"],"external":["pilot_charter_approved"],"provides":["evidence.pilot"],"uses":["engine.running"],
   "read_scope":["governance/campaigns/AGENT-OPS-X/contracts/benchmark.md"],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":[],"ac04":12},
  {"id":"W4-P03","file":"packets/W4.md","class":"D","pub":"M0","executor":"engine","deps":["W3-P10"],"external":[],"provides":["evidence.drills"],"uses":["engine.running"],
   "read_scope":[],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.igx_gov wave drill all"],"ac04":0},
  {"id":"W4-P04","file":"packets/W4.md","class":"St","pub":"M0","executor":"engine","deps":["W3-P10"],"external":["oa3_applied"],"provides":["evidence.security"],"uses":["engine.running","ao.probe.github"],
   "read_scope":[],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.agent_ops.probe github"],"ac04":1},
  {"id":"W4-P05","file":"packets/W4.md","class":"St","pub":"M0","executor":"engine","deps":["W4-P01","W4-P02"],"external":[],"provides":["evidence.quality"],"uses":["engine.running"],
   "read_scope":[],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.igx_gov doctor"],"ac04":1},
  {"id":"W4-P06","file":"packets/W4.md","class":"D","pub":"M0","executor":"engine","deps":["W4-P03","W4-P04","W4-P05"],"external":[],"provides":["evidence.kpi"],"uses":["igx_gov.usage"],
   "read_scope":[],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":["uv run python -m scripts.governance.igx_gov usage report --campaign AGENT-OPS-X"],"ac04":0},
  {"id":"W5-P01","file":"packets/W5.md","class":"EC","pub":"M0","executor":"auditor","model":"opus/high","deps":["W4-P06"],"external":[],"provides":["evidence.g6"],"uses":["evidence.kpi"],
   "read_scope":["governance/campaigns/AGENT-OPS-X/**"],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":[],"ac04":0},
  {"id":"W5-P02","file":"packets/W5.md","class":"var","pub":"var","executor":"engine","deps":["W5-P01"],"external":[],"provides":["corrective.done"],"uses":["evidence.g6"],
   "read_scope":["governance/campaigns/AGENT-OPS-X/evidence/g6-audit.md"],"edit_scope":["@corrective"],"test_commands":["@corrective"],"ac04":0},
  {"id":"W5-P03","file":"packets/W5.md","class":"EC","pub":"M0","executor":"auditor","model":"opus/high","deps":["W5-P02"],"external":[],"provides":["evidence.g6_delta"],"uses":["corrective.done"],
   "read_scope":["governance/campaigns/AGENT-OPS-X/**"],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],"test_commands":[],"ac04":0},
  {"id":"W5-P04","file":"packets/W5.md","class":"D","pub":"M2","executor":"engine","deps":["W5-P03"],"external":[],"provides":["activation.prepared"],"uses":["evidence.g6_delta"],
   "read_scope":[],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**","governance/campaigns/AGENT-OPS-X/activation.md","governance/agent-control-plane/legacy_tickets.txt"],
   "test_commands":["uv run python -m scripts.governance.igx_gov charter verify AGENT-OPS-X"],"ac04":0},
  {"id":"W5-P05","file":"packets/W5.md","class":"D","pub":"M0","executor":"engine","deps":["W5-P04"],"external":["w5_pr_merged"],"provides":["activation"],"uses":["activation.prepared"],
   "read_scope":[],"edit_scope":["governance/campaigns/AGENT-OPS-X/evidence/**"],
   "test_commands":["uv run python -m scripts.governance.igx_gov charter verify AGENT-OPS-X","uv run python -m scripts.governance.igx_gov wave activate AGENT-OPS-X"],"ac04":0}
 ],
 "corrective_rules": {"path":"packets/corrective/W5-C*.md","edit_scope":"must be a subset of the union of edit_scope of the packets cited by the finding","class":"computed by routing, never below the cited packets","budget":"W5 allocation only","tests":"cited packets' test_commands plus a regression test for the finding"}
}
```
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/bootstrap.md -->
# Contract — bootstrap transition protocol (W0–W3)

<!-- igx:immutable begin -->
**BT-0 — W0-P00.**
- The Operator completes OA-1 and starts the bootstrap executor as `igx-agent`:

  ```
  claude --permission-mode dontAsk --add-dir D:\Projekty\_wt\intergrax\_agent\inbox
  ```

  It runs in the agent clone, with the W0-P00 user settings and start prompt (`/goal`).
- The publisher is **not** running yet.

**BT-1 — from W0-P01.** Each turn is a loop, using only commands the W0-P00 allow list permits:
1. `uv run python -m scripts.governance.agent_ops.boot next`. Before `boot.py` exists, read W0-P01 from the inbox plan with `rg -n` plus Read with an offset.
2. Read that packet's file and its `packets.md` entry.
3. `boot budget check <id>`. W0-P01 to W0-P03 use a fixed allowance of $8 until W0-P04 delivers the check.
4. Implement it, run its `test_commands`, and commit with the trailers `Ref:` and `Packet:`.
5. `boot integrate <id>`, then `boot done <id>`. This writes `.git/igx-ops/boot.json` and `boot-handoff.md` (≤ 1 KB).
6. `boot publish-wave`. It writes a spool request only when every precondition holds:
   - all packets of the wave are `implemented`
   - the previous wave is merged
   - `probe isolation` is PASS
   - `probe workflow-preflight` is PASS (after OA-3a, if it was required)
   - `boot validate` is green
   - from W2-P02 on, the local `gate.py` run is green

   Otherwise it records `waiting_external`.
7. Go to the next packet.

After a compaction or restart, read only `boot-handoff.md` and `boot status`.

**BT-1a — first publication (W0).**
- `boot publish-wave W0` finds no running publisher, records `waiting_external: oa2a_publisher_running`, and continues with W1 packets.
- After OA-2a, the publisher runs `selftest`, then processes the W0 request.

**BT-2 — W2.**
- W2-P08 runs once `w0_pr_approved` holds.
- OA-3 is requested after `w2_pr_merged`. Until then, W3 continues.

**BT-3 — W3-P01.** `state import-boot` imports `boot.json` and the bootstrap ledger, which includes every `igx-agent` transcript since W0-P01.

**BT-4 — W3-P10.**
1. The executor runs:

   ```
   hc run --bg --no-notify --outlive-parent --name ao-engine -- uv run python -m scripts.governance.igx_gov wave run AGENT-OPS-X
   ```

   This runs from the engine worktree, pinned at the `agent-ops/w3` tip.
2. It confirms `engine_lock`.
3. It ends the session. The stop guard allows this because the engine is running.

From then on, only the engine executes.

**Never in bootstrap:**
- `git push`, `gh`, or direct GitHub API calls. The spool is the only path to GitHub.
- Edits outside `charter.edit_globs`.
- Go-ahead questions.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/operator-actions.md -->
# Contract — consolidated Operator actions (complete list)

<!-- igx:immutable begin -->
## OA-1 — W0-P00, once (about 75 min)

Work through the checklist in order. Each step's verification runs later, in W0-P02 and the publisher selftest.

**Accounts and the GitHub App**

1. Create two local **standard** Windows users, `igx-agent` and `igx-pub`. Neither may be an administrator.
2. In your GitHub account, go to Settings → Developer settings → GitHub Apps → New.
   - Name: `igx-agent-<suffix>`.
   - Webhook: off.
   - Permissions: Contents **Read and write**, Pull requests **Read and write**, Metadata **Read**. Nothing else.
   - "Only on this account".
   - Install it on this repository only.
   - Generate the private key and move it to `C:\Users\igx-pub\.igx\app.pem`. Set the ACL to `igx-pub` only (plus SYSTEM and Administrators), and delete your downloaded copy.
   - Write `C:\Users\igx-pub\.igx\app.json` as `{"app_id":…, "installation_id":…, "app_slug":"…"}`.

**`igx-agent` toolchain and clone**

3. As `igx-agent`, install `uv`, Python 3.12, Node, `npm i -g @anthropic-ai/claude-code@2.1.329`, and herdr/hc.
   - Run `claude` and log in with your Max subscription.
   - Confirm that no `ANTHROPIC_*` variable is set.
4. As `igx-agent`, clone the repository anonymously over HTTPS into `D:\Projekty\_wt\intergrax\_agent\repo`.
   - Create the folders `D:\Projekty\_wt\intergrax\_agent\{wt,inbox,spool\req,spool\res}`.

**`igx-pub` toolchain and trusted clone**

5. As `igx-pub`, install Python 3.12 (and `uv`, used only to create the venv).
   - Clone the repository into `C:\Users\igx-pub\trusted\repo`. That profile is private by default.
   - Do **not** run `uv sync`, and never install the repository's project dependencies. The publisher venv comes from the hash-pinned lock in OA-2a.

**Folder permissions**

6. Set these ACLs:

   | Path | Rights |
   |---|---|
   | `_agent\repo` | `igx-pub` read |
   | `spool\req` | `igx-agent` write, `igx-pub` read |
   | `spool\res` | `igx-pub` write, `igx-agent` read |
   | `spool` (root, for the `PAUSE` file) | you write; `igx-pub` read and write; `igx-agent` **read only** |
   | `_agent` | you read |
   | `D:\Projekty\intergrax` | **no** rights for `igx-agent` or `igx-pub` |

**Plan and `igx-agent` settings**

7. Copy this plan to `...\_agent\inbox\AGENT-OPS-X-plan-v2.1.md`.
8. Write `C:\Users\igx-agent\.claude\settings.json` from the block in `packets/W0-P00.md`.
   - In `C:\Users\igx-agent\.config\herdr-core\config.toml`, set `notes = false`.

**Repository settings**

9. Set Actions → General → Workflow permissions → **Read repository contents**.
10. Commit `.github/CODEOWNERS` directly to `development` from your own checkout. Its content: `* @<your-login>`. Do this **before** step 11.
11. Edit the ruleset "Protect development" to the bootstrap configuration:
    - Require PR, 1 approval
    - Dismiss stale approvals
    - Require review from Code Owners
    - Require approval of the most recent push
    - Block force pushes and deletions (keep these)
    - No bypass list

**Your own settings**

12. Keep "extra usage" disabled in claude.ai.

**Start the bootstrap session**

13. As `igx-agent`, in `_agent\repo`, run:

    ```
    claude --permission-mode dontAsk --add-dir D:\Projekty\_wt\intergrax\_agent\inbox
    ```

    Paste the start prompt from `packets/W0-P00.md`.

## OA-2a — once, when `boot status` shows `waiting_external: oa2a_publisher_running` (about 15 min)

**Review the whole trusted runtime chain, not only `publisher.py`.** The review package is the folder `D:\Projekty\_wt\intergrax\_agent\repo\scripts\governance\agent_ops\trust_chain\` on the W0 wave branch, with a generated `REVIEW.md` that lists every file with its SHA-256 and line count.

| File | Limit | What to check |
|---|---|---|
| `publisher.py` | ≤ 350 lines | stdlib + pyjwt + cryptography only; typed spool requests only; never runs anything inside the agent clone; never prints a token |
| `run_publisher.py` (the launcher) | ≤ 80 lines | starts only the pinned publisher; honours `PAUSE`; calls the update logic below |
| `update_trusted.py` | ≤ 120 lines | the trusted-revision update rule below |
| `publisher.lock` | — | hash-pinned, wheels-only lock of exactly `pyjwt`, `cryptography` and their dependencies. Check each package against the official index. |
| `trust_chain.json` | — | the complete list of files whose change needs the Operator's approval: the four files above plus `merge.py`, `gate.py`, `charter.py`, `classify.py`, `risk_map.json`, `agent_ops.json` |

**Steps**

1. Review the table above. If anything is doubtful, do nothing and raise E1 in the agent session.
2. Copy the reviewed files to `C:\Users\igx-pub\trusted\bootstrap\`, and copy `run_publisher.py` also to `C:\Users\igx-pub\trusted\run_publisher.py`.
3. As `igx-pub`, create the venv from the lock:
   ```
   python -m venv C:\Users\igx-pub\trusted\venv
   C:\Users\igx-pub\trusted\venv\Scripts\pip install --require-hashes --only-binary :all: -r C:\Users\igx-pub\trusted\bootstrap\publisher.lock
   ```
4. Write `C:\Users\igx-pub\trusted\pin.json` as `{"sha":"<development HEAD of the trusted clone>"}`.
5. Create the Scheduled Task "IGX-Publisher":
   - Run as `igx-pub`, "whether user is logged on or not".
   - Triggers: at startup, plus now. Restart on failure every 1 min.
   - Action: `C:\Users\igx-pub\trusted\venv\Scripts\python.exe C:\Users\igx-pub\trusted\run_publisher.py`.

**Trusted-revision update rule** (implemented in `update_trusted.py`; changing the rule is itself a trust-chain change). The launcher advances `pin.json` and the trusted clone only if, for every commit between the pin and the new `development` HEAD, **each change to a file listed in `trust_chain.json` comes from a merged PR that has the Operator's `APPROVED` review at that PR's head SHA** (read through the GitHub API). Otherwise it stays on the pin, writes `PAUSE.auto` and raises E5. The launcher `run_publisher.py` lives outside the repository and is updated only by you.

Your approval of the W0 PR at its head counts as the approval of the chain for the files in that PR. The first update therefore passes only if the merged files are the ones you reviewed here, or a later change that you approved.

## OA-2 — asynchronous approvals (code-owner reviews)

These are the only approval clicks:
- the W0 PR (this is the charter approval)
- the W1 PR
- the W2 PR
- the W3 PR, together with the W4 pilot charter PR
- the W5 PR (activation)

The W4 evidence PR (M0) merges automatically after OA-3.

## OA-3 — after `w2_pr_merged`, a maintenance window of about 20 min

0. **Pause the publisher.** Create the empty file `D:\Projekty\_wt\intergrax\_agent\spool\PAUSE`, then wait until `spool\res\status.json` shows `"state":"paused","in_flight":0`. The agents keep working, because publication waits as `waiting_external`.
1. Set the "Protect development" ruleset enforcement to **Disabled**.
2. From your checkout, commit directly to `development` the files `evidence/patches/agent-ops.yml` (as `.github/workflows/agent-ops.yml`) and `evidence/patches/CODEOWNERS`. The workflow then runs once on push, which registers the checks.
3. Edit the ruleset per `governance/agent-control-plane/github/rulesets/development.json`:
   - 0 required approvals
   - Code Owners required
   - Dismiss stale approvals
   - Required checks `agent-ops-gate`, `agent-ops-tests` and `ci-smoke`, each with source **GitHub Actions**, "up to date" on
   - Block force pushes and deletions
   - No bypass actors
   - Set it to **Active**
4. Apply `main.json`: add `agent-ops-gate` to main.

5. **Resume the publisher.** Delete `spool\PAUSE`. The publisher runs `restore-verify` (rulesets, workflow, CODEOWNERS, workflow preflight, selftest) and resumes only if it passes. If it fails, it pauses itself again and raises E5 with the failing check.

`probe github` then verifies the configuration, and `boot` records `oa3_applied`.

## OA-3a — only if the W0-P02 workflow preflight reports a FAIL

Apply `evidence/patches/workflow-guard.md` in your own commit **before** OA-2a.

## OA-4 — only on `ESCALATION E1–E8`

Choose one of the options presented.

## OA-5 — only after a Windows reboot

As `igx-agent`, run:

```
uv run python -m scripts.governance.igx_gov wave resume AGENT-OPS-X
```

The publisher restarts on its own.

## OA-6 — optional

Apply `evidence/operator-profile-patch.md`.

## OA-7 — break-glass, only if the gates deadlock every PR

You alone (as admin):
0. Create `spool\PAUSE` and wait for `"state":"paused","in_flight":0`.
1. Disable the ruleset.
2. Commit the fix.
3. Re-enable the ruleset, in the state defined in `governance/agent-control-plane/github/rulesets/*.json`.
4. Record the event in `evidence/break-glass.md`.
5. Delete `spool\PAUSE`. The publisher resumes only after `restore-verify` passes. Until then nothing is published.

Agents never have this right.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/escalation.md -->
# Contract — autonomy and escalation

<!-- igx:immutable begin -->
**Never stop for, and never ask about:**
- a finished ticket, packet, step or wave
- running tests
- fixing defects within scope
- updating a status
- creating a worktree or branch
- commit, integration or publication requests
- starting a review
- moving to the next packet or the next approved wave
- reading a file
- launching an allowed worker
- retrying within limits
- `waiting_external`

**Escalate only for:**

| Code | Trigger |
|---|---|
| E1 | A decision that is not in the ledger |
| E2 | A contradiction |
| E3 | An edit outside the globs, or a packet that is not in the DAG |
| E4 | A gate still red after the loop limit, the same failure signature twice, or a failed post-activation health check |
| E5 | A P0 security, data-loss or secret-exposure finding |
| E6 | Missing access or resource |
| E7 | An action that is not pre-authorized |
| E8 | Budget breach, or no progress while spending grows |

**Protocol:**
1. `escalate` blocks only the dependent packets.
2. Send one batched message that starts with `ESCALATION E<n>:`. For each item give the problem, options, a recommendation, the consequence and the cost.
3. Record the Operator's answer in the ledger and run `resolve`.

**Reports:** one at the end of each wave and one at the end of the campaign, each ≤ 40 lines.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/routing.md -->
# Contract — adaptive routing matrix

<!-- igx:immutable begin -->
| Class | Criteria (max wins, never downgraded) | Executors | Model / effort | maxTurns | Reservation per attempt | Loops | Review rounds |
|---|---|---|---|---|---|---|---|
| D | `test_commands` / run fully specify the work | engine / boot | none | — | $0 | — | — |
| S | ≤ 3 files; one tier; no M2S or M3; ≤ 1 test command | implementer | sonnet / medium (haiku / low for read-only research) | 40 | $1.5 | 2 | 0 |
| St | ≤ 10 files; one domain; M1 or M2 | implementer + reviewer | sonnet / medium | 80 / 25 | $4 / $1.5 | 2 | 2 (delta) |
| C | > 10 files, cross-module, new contract, state, security, or M2S | implementer + reviewer (+ consult after 2 failed loops) | implementer sonnet / high; reviewer opus / high for M2S, otherwise sonnet / medium; consult opus / high, ≤ 40 lines | 120 / 30 / 15 | $8 / $4 / $3 | 3 | 3 |
| EC | Certification, IIAC/FRZ gates, Enterprise M3, G6 | as C + qualification tests + separate G6 auditor | as C; auditor opus / high | as C | as C + $15 | 3 | 3 + G6 |

**Model choice:**
- Opus only for pre-approval architecture, W0-P06, consults, M2S/M3/EC reviewers, and G6.
- Sonnet by default.
- Haiku for bounded research.
- No model for tests, CI, status or scheduling.

**Workers:**
- Each starts as a fresh `claude -p` session.
- `--disallowedTools Agent`: workers cannot spawn subagents.
- No GitHub access: there is no token in their environment, and the publisher key sits in another Windows account.
- `context: fork` requires a reason.
- Interactive sessions deny `Agent(general-purpose)`.

**Context per role:**

| Role | Receives |
|---|---|
| Implementer | packet + `read_scope` + digest (≤ 1.5 KB) |
| Reviewer | diff bundle + acceptance + delta |
| Consult | digest + packet |
| Auditor | closure matrix + evidence |
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/budget.md -->
# Contract — budget controller with atomic reservations

<!-- igx:immutable begin -->
## Units

- Est. USD at list price, labelled "API-equivalent estimate; not billing".
- Tokens tracked by category.

## Ledger

The ledger is append-only. It covers attempt → session → packet → ticket → wave → campaign.

| Row kind | Meaning |
|---|---|
| `reserve` | budget held for a launch |
| `settle` | actual spend recorded |
| `release` | unused reservation returned |
| `transfer` | budget moved between scopes |
| `charge_full` | attempt charged at its full reservation |

## Reservation

A reservation is made inside a single `BEGIN IMMEDIATE` transaction, fenced by the engine epoch. It succeeds only if, at every scope level:

```
settled + Σ(active reservations) + new + margin ≤ cap
```

- `margin` = 10 % of active reservations + $2 per bound interactive session not yet imported.
- For every non-W5 scope, `cap` is reduced by the remaining W5 reserve: `max(0, 60 − W5_settled)`.

## Settlement

- **Normal attempt:** settle from `runs/<attempt>.json`, then release the remainder.
- **Orphaned attempt:** stays held until it can be settled from the transcript, found by its `--session-id`.
- **Unresolvable attempt:** `charge_full`.
- **Interactive campaign sessions:** imported on every scheduler tick.

## Thresholds

| Level | Action |
|---|---|
| < 70 % | normal |
| 70–90 % | delta-only review, trimmed digests, concurrency 1 |
| 90–100 % | finish running attempts; no new reservations |
| ≥ 100 % | E8 |

## No progress

An attempt counts as making no progress if it spends more than 50 % of its reservation with no diff change, or repeats the same failure signature.

## Enforcement categories

| Category | Limits |
|---|---|
| 1 — enforced locally | turns, attempt spend (client estimate), context window, output size, worker and subagent counts, loops, repeated reads, reservation check |
| 2 — delayed | settled totals, interactive-session import, rate limits |
| 3 — advisory | interactive rotation |
| 4 — report only | Anthropic server-side and subscription accounting |

## Integrity

`igx-agent` processes can write the ledger file. Ledger integrity is therefore **process-level**. Independent cross-checks are the transcripts and `rate_limits`, compared in the W4-P06 report.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/publication.md -->
# Contract — publication, publisher, server-side enforcement

<!-- igx:immutable begin -->
## Classes

Classes come from `classify.py` plus `governance/agent-control-plane/risk_map.json` (explicit files plus content signals).
- The highest class wins.
- An unknown path counts as M2.
- **The declared class must be ≥ the computed class.** The required approvals always follow the computed class.

| Class | Typical content | What merge requires |
|---|---|---|
| M0 | status, evidence, non-qualification docs | server-side gates |
| M1 | instructions without `igx:authority`, dev tooling outside gate/hooks/authority | gates + review record |
| M2 | settings, hooks, agent-control-plane, pin, ref-config, governance tests, qualification docs, ADR, `igx:authority` | Operator approval at the head |
| M2S | `ao/{classify,gate,charter,merge,publisher}.py`, `risk_map.json`, `agent_ops.json`, `.github/**`, rulesets | Operator approval at the head; workflows and CODEOWNERS are applied only by the Operator |
| M3 | product tiers | existing Operator requirements |

**Content signals that raise a PR to ≥ M2:**
- permission or hook keys
- `INTERGRAX_SKIP_*`
- `--no-verify`
- `continue-on-error`
- `Approved:` or `Charter:` lines
- required-check names
- `pull_request_target`
- credentials

## Publisher (`ao/publisher.py`, runs as `igx-pub`)

**Typed spool requests** (JSON in `spool/req`, each ≤ 64 KB):

| Request | Constraint |
|---|---|
| `selftest` | — |
| `push{branch}` | the branch must match `^agent-ops(-bench)?/[a-z0-9._/-]+$` |
| `pr_upsert{branch,title,body}` | — |
| `pr_ready{pr}` | — |
| `comment{pr,kind,body}` | `kind` is `review_verdict` or `report` |
| `merge{pr}` | — |
| `close{pr}` | only `agent-ops-bench/*` |
| `delete_branch{branch}` | only `agent-ops/probe-*` |
| `poll{pr}` | — |

Every other request is rejected. Results are written to `spool/res`.

**How it executes:**
- **Fetch:** it fetches matching refs from the agent clone into its trusted clone, under `refs/remotes/agent/*`. It never runs git commands, hooks, config or code inside the agent clone.
- **Push:** from its own clone, using its own config, with a 1 h App token narrowed to Contents and PR.
- **Merge:** it runs `merge.py` preconditions **from the trusted clone** (development HEAD), then `gh pr merge --merge --match-head-commit <sha>`. After that it does `git pull --ff-only` in the trusted clone.
- **Preflight:** before the first publication and before every push batch it re-runs `workflow-preflight` on the trusted clone. Anything other than PASS → refuse and raise E5.
- **`selftest` checks**, each must hold:
  - App permissions are exactly as specified
  - `current_user_can_bypass == never` on every ruleset
  - pushing scratch branch `agent-ops/probe-<ts>` succeeds
  - pushing a commit that adds `.github/workflows/_probe.yml` (`on: workflow_dispatch` only, harmless) is **rejected**
  - posting a commit status is **rejected**
  - the scratch branch is deleted afterwards

**Fallback:** if selftest shows the App path does not work, raise E6 and use the OA-1 fallback (a machine user with a classic `repo` PAT, no `workflow` scope), also run only inside the publisher.

## Publisher pause (OA-3 and every break-glass)

- **Control files:** `spool\PAUSE` (written by the Operator) and `PAUSE.auto` (written by the publisher). Either one pauses publication. `igx-agent` can only read them.
- **While paused:** the publisher handles only `poll` and `selftest`, and writes `spool\res\status.json` with `"state":"paused"`. The engine and `boot.py` treat it as `waiting_external: publisher_paused`. Agent work does not stop and nothing is asked of the Operator.
- **Idle confirmation:** the Operator waits for `"state":"paused","in_flight":0` before changing any protection.
- **Automatic pause:** the publisher pauses itself (`PAUSE.auto`) when a protection is missing: a ruleset that is disabled, changed or has a bypass actor, a changed workflow or CODEOWNERS file on `development`, a failing `workflow-preflight`, or a failing trusted-revision update.
- **Resume:** removing `PAUSE` is not enough. When no pause file is left, the publisher first runs `restore-verify` and resumes only if it passes. `restore-verify` checks:
  - every ruleset is Active and matches `governance/agent-control-plane/github/rulesets/*.json`, with required checks pinned to GitHub Actions, the strict policy on, 0 bypass actors, and Code Owners on
  - `.github/workflows/agent-ops.yml` and `.github/CODEOWNERS` on `development` match the approved patches
  - `workflow-preflight` is PASS
  - `selftest` passes
  - the trusted revision follows the pin rule
- **On failure:** `PAUSE.auto` is written again and E5 is raised with the failing check.

## Ruleset bootstrap order

1. **OA-1:** commit CODEOWNERS `* @operator` first, then the bootstrap ruleset (PR, 1 approval, Code Owners, stale dismissal, last-push approval, no bypass). The Operator never needs to approve their own PR, because the Operator's own changes are made in maintenance windows (OA-3 and OA-7) and never through a PR.
2. **W0–W2:** every PR is authored by the App and approved by the Operator, so the "author cannot approve" rule never applies. The publisher merges after approval.
3. **OA-3,** after the W2 PR (with the gate code) has merged, with the publisher paused:
   0. Create `spool\PAUSE` and wait for `"state":"paused","in_flight":0`.
   1. Disable the ruleset.
   2. The Operator commits the workflow and the narrowed CODEOWNERS. The workflow runs on push to `development`, which registers the checks so their source can be pinned.
   3. Apply the final ruleset.
   4. Re-enable it.
   5. Remove `spool\PAUSE`. The publisher runs `restore-verify`, and publication resumes only when it passes.
4. **M0/M1 auto-merge** is possible because the final ruleset has **0 required approvals**, and the narrowed CODEOWNERS owns only M2/M2S paths. No code-owner review is required for unowned files. Last-push approval is **off** in the final ruleset, because the gate's head-SHA approval check covers M2+.

   This is proven in W4: the W4 evidence PR (M0) must merge without any approval. If it doesn't → E4.
5. **Deadlock with no bypass actors:** use OA-7 break-glass, which is the Operator's admin action only, performed with the publisher paused.

## Server-side conditions

None of these depend on the engine or on `igx_gov merge`.

| Condition | How it is enforced |
|---|---|
| Head SHA | Checks are per SHA. Stale approvals are dismissed. The gate counts only approvals with `commit_id == head`. |
| Base is current | The strict up-to-date policy. |
| Mandatory tests | The `agent-ops-tests` job runs the approved packets' `test_commands`. The test list is read from `contracts/packets.md` only after `charter.verify` passes in the trusted job. `ci-smoke` must also pass. The W0-P05 trigger analysis confirms that every required check actually runs on PRs to `development`, otherwise it raises E2. |
| Operator authorization (M2+) | CODEOWNERS from base, plus the gate's check of the Operator's `APPROVED` review at the head for content-signal M2. The gate re-runs on `pull_request_review`. |
| No self-modification | The gate runs from a base checkout. The App lacks Workflows permission, so it cannot change or add any workflow, and so cannot replace a required check with a same-named one. Checks are pinned to GitHub Actions, so App statuses are impossible and fallback statuses are ineffective. CODEOWNERS is read from base. |
| No elevated untrusted code | Only `pull_request` triggers (never `pull_request_target`); `permissions: {contents: read, pull-requests: read}`; no secrets. `agent-ops-gate` runs only base code under `python -I` with `sys.path` limited to `_trusted`. `agent-ops-tests` runs PR code in a separate job and runner, with read-only access. |
| ACT-N11 | The gate runs the base IIAC verifier over the PR tree as data, and fails only on **new** errors versus `evidence/act_n11_baseline.json`. Existing gates are unchanged. |
| Direct API use | LLM processes have no GitHub credential. The publisher accepts typed requests only. Even a direct API merge by the App identity is still subject to every rule above. |

## Residual risk (the real boundary)

**Guaranteed:**
- Windows account separation: `igx-agent` cannot read the Operator's or `igx-pub`'s profiles, or the trusted clone. Verified in W0-P02 and W4-P04.
- The publisher runs only code from operator-approved commits (bootstrap: the single file the Operator reviewed in OA-2a).
- GitHub server-side rulesets, CODEOWNERS, Actions-pinned checks, and App permissions without admin, workflows or statuses.

**Not guaranteed:**
- **Local administrator or OS privilege-escalation compromise.** Out of scope; that is the Operator account.
- **M0/M1 review independence.** An `igx-agent` process could submit a forged `review_verdict` request. This is process-level (L2). Its impact is limited to M0/M1 content, which by classification cannot touch authority, hook, gate or settings paths.
- **Budget ledger and engine state integrity.** Writable by `igx-agent`, so process-level. Cross-checked against transcripts.
- **Content of PRs proposed by agents.** Always visible, and gated as above.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/authorization.md -->
# Contract — immutable authorization (canonical format, fail-closed)

<!-- igx:immutable begin -->
## Canonical format of an `immutable` file

1. **Line 1:** exactly one Markdown H1 line, `# …`.
2. **Then:** zero or more blank lines.
3. **Then:** exactly one immutable block, delimited by the HTML comments `igx:immutable begin` and `igx:immutable end`, each on its own line. The H1 and the block content form the **immutable content**. Only the immutable content is hashed and compared.
4. **Then:** optionally, mutable blocks, delimited by the HTML comments `igx:mutable begin` and `igx:mutable end`. These are allowed **only in `charter.md`**. They are checked for shape only. Their content is excluded from every digest and from the `split(source)` equality check, and may differ from the source.
5. **Then:** only blank lines.

Marker comments must not appear anywhere else in a file, including when a document quotes them.

Anything else outside these places → reject. Normalization before hashing: LF line endings, trailing whitespace stripped, UTF-8 without BOM.

## File kinds

The authorized file set is derived deterministically, not trusted from the manifest. Every file under the campaign directory must match exactly one kind; anything else → reject.

| Kind | Files | Rule |
|---|---|---|
| `immutable` | every file produced from the file markers of `source/plan-v2.1.md` | its **immutable content** must equal that of `split(source)` byte for byte; the mutable blocks of `charter.md` are compared by shape only |
| `source` | `source/plan-v2.1.md` | its digest must equal the `Plan digest:` of tickets A–F |
| `manifest` | `manifest.json` | never trusted as input. `verify` recomputes it and requires equality. |
| `generated` | `contracts/benchmark.md` (from W0-P06) and `packets/corrective/W5-C*.md` (from W5-P02) | **content rules are re-checked on every `verify`** (`bench-check` for the benchmark; `corrective_rules` in `contracts/packets.md` for corrective packets), and its digest must be listed in `generated.lock` |
| `ledger` | `generated.lock` (in the campaign directory, **not** in `evidence/`) | append-only list `{path, sha256, created_in_commit}`. Each entry must match its file, and each file may be listed once. Its content must be a superset of the version at the last Operator-approved commit, with only appended entries. |
| `evidence` | `evidence/**` | mutable, never authority |
| `derived` | `approval.json` | a cache only; recomputed from the GitHub API on every `verify` |
| `revocation` | `REVOKED` | if present, the approval is revoked. Only the Operator creates it. Deleting it requires a code-owner PR. |
| `activation` | `activation.md` | a declaration written by W5-P04, effective only on merge of the W5 PR with the Operator's approval at head. The factual activation is recorded by W5-P05 in `evidence/activation-record.md`. |

## Why swapping a generated file and its digest does not bypass authorization

A generated document and its digest are valid only if **all three** hold:
1. **Content rules pass when recomputed.** A hand-made replacement that violates the scope, class or budget rules fails even with a matching digest.
2. **Operator-approved anchor.** `contracts/**`, `packets/**`, `generated.lock` and `manifest.json` are M2S/M2 paths in `risk_map.json`. Any change to them needs the Operator's code-owner approval at the head SHA (server-side gate and CODEOWNERS). The agent cannot publish a replaced file with a replaced digest without that approval.
3. **Append-only ledger.** An entry cannot be edited or removed after approval, and a changed digest for an existing path is rejected.

Corrective packets are executed under the pre-authorization A-10 and must satisfy the content rules at execution time. They are published inside the W5 PR, which the Operator approves at head.

## `verify` rejects on

- a ledger violation (see above)
- an unknown or missing file
- a canonical-format violation
- `split` mismatch
- source digest mismatch
- an incomplete digest input set
- a generated digest change
- `REVOKED`
- expiry
- an approval tuple mismatch

## Immutable digest

SHA-256 over the immutable content of all `immutable` files, in sorted path order.

## Approval tuple

`(campaign_id, immutable_digest, approved_commit_sha, review_id, approver_login)`

- Re-fetched on every `verify`.
- Requires state `APPROVED`, `commit_id == approved_commit_sha`, a tree at that commit that yields the digest, and approver = `operator_login`, which is never the App or machine user.
- **Repeated verification of the same active tuple is idempotent and valid.** It is not replay.
- **Replay** is any of:
  - the same `review_id` combined with a different digest, campaign or commit
  - use after `valid_until` (120 days after `approved_at`)
  - use after revocation (`REVOKED`, or the review dismissed)

## `amend`

Allowed only on `igx:mutable` blocks in the charter. Anything else → E1/E3.

## Required tests (must fail)

- edit a contract's immutable text
- an extra or unknown contract file
- remove the manifest or a file
- strip markers, or add text outside a block
- change the DAG JSON (deps, class, pub, scopes, tests)
- raise a budget
- loosen publication
- remove a test command
- widen a glob
- change `benchmark.md` or a corrective packet after its freeze, with and without updating `generated.lock`
- edit or remove an existing `generated.lock` entry
- replace a generated file and its ledger entry in one PR without Operator approval
- a corrective packet outside its cited scope
- replay
- an old commit
- approval by the App

## Required tests (must pass)

- a mutable charter edit
- re-verify the same approval
- a valid corrective packet
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/state.md -->
# Contract — durable execution state and recovery

<!-- igx:immutable begin -->
## Store

`<agent clone>/.git/igx-ops/state.db`: SQLite in WAL mode with a 10 s busy timeout.

## Tables

- `campaigns`, `waves`
- `packets`, loaded from the `contracts/packets.md` JSON
- `attempts`: role, worker_id, session_id (from `--session-id`), worktree, model, effort, usage, cost, failure_sig, checkpoint
- `leases`: packet, attempt, pid, hc terminal, expires_at, epoch
- `engine_lock`: pid, epoch, heartbeat_at
- `idempotency`, `budget_ledger`, `escalations`, `events`

## Packet statuses

`pending`, `ready`, `leased`, `implementing`, `verifying`, `reviewing`, `implemented`, `publishing`, `published`, `waiting_external`, `blocked_escalation`, `failed`, `stalled`.

## Fencing

- Taking over a lock whose heartbeat is older than 2× the interval increments `epoch`.
- Every write transaction checks `epoch` inside `BEGIN IMMEDIATE`. On a mismatch the old engine aborts.
- Before any side effect (a launch or a spool request), the engine checks the epoch. The request carries an idempotency key `pub:<packet>:<op>:<sha>`, and the publisher deduplicates on it.
- The new engine adopts or kills the old engine's workers.

## Rebuild without the DB

| What is rebuilt | Source |
|---|---|
| Attempts and spend | `runs/<attempt>.launch.json` (written before launch) and `runs/<attempt>.json`, plus transcripts matched by session_id. An attempt that can't be resolved → `charge_full`. |
| `implemented` | `Packet:` trailers on the wave branches |
| `published` | GitHub facts, through the publisher's `poll` and the `spool/res` history |
| Escalations | `evidence/escalations.md` |

## Backups

A backup is taken at each wave transition. Restore = backup + rebuild for everything after it.

## Guarantees

- Settled budget is never re-allocated.
- Nothing is published twice.
- `implemented` or `published` work is never redone.

## Test cache

- Key: `(tree hash, command, uv.lock hash, python version, OS, env-allowlist hash)`.
- Only commands listed as `cache: deterministic` in `agent_ops.json` are cached: ruff, and unit pytest without network markers.
- Anything depending on time, network, GitHub, CI or an external service is `never` cached.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/stop-guard.md -->
# Contract — state-driven Stop guard

<!-- igx:immutable begin -->
**Source:** `igx_gov wave status --session <id> --json`, or `boot status` before W3. The transcript text is never the source.

| State | Decision |
|---|---|
| Unbound, or the campaign is `done_verified` | allow |
| A `ready` step is assigned to this session | **block**: "Campaign approved; next: <step>. Continue; stop only with ESCALATION E<n>." |
| The engine is running | allow |
| `waiting_external` | allow |
| An escalation is blocking this session | allow |
| The state is invalid | allow + event + reconcile |
| Budget ≥ 100 % | allow + E8 |

**Loop limit:** on the third consecutive block for the same step, allow the stop and mark the step `stalled_interactive`. The engine then reassigns it to a headless worker. The native cap of 8 blocks is the outer bound.

**Session start:** SessionStart (all sources) injects at most 10 lines of `ctx`.

**`last_assistant_message`:** used for diagnostics only, to count AC-03 phrases.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/contracts/measurement.md -->
# Contract — measurement methodology

<!-- igx:immutable begin -->
## Sources

- `igx-agent` transcripts (all W4 runs, both legacy and new, happen in that profile)
- headless JSON results
- `rate_limits`
- optional `npx ccusage@<pin> --offline`

The baseline is the 2026-11-20 measurement (annex H, method validated).

## Counting rules

- Deduplicate by `message.id` per file, plus a global check.
- Split cache-write into 5m and 1h.
- Thinking comes from `output_tokens_details`.
- Exclude `<synthetic>`.
- Count compactions and cache rebuilds.
- Keep separate baselines for main sessions and subagents.
- Use a pinned cutoff.
- Flag turns above 200k.

## Unit of comparison

One task run, from the first launch to publish-ready (acceptance tests pass, gates green, review PASS). It includes failed attempts, reviews, consults, overhead, and legacy resumes. It excludes CI compute and human time.

**Legacy runs** are headless, at the baseline SHA with legacy rules. Each go-ahead stop is resumed with `--resume <id> "kontynuuj"` and counted as one continuation question. Wall time excludes Operator latency, which favours legacy.

## Failure

A run that does not reach acceptance within 2× its class reservation is a failure. Its full cost counts. If the new system's success rate is below legacy's, that class fails AC-01.

## Aggregation

| Metric | Per-class ratio | Overall |
|---|---|---|
| AC-01a, AC-01b, AC-02 | median(new) / legacy | 1 − geometric mean of the ratios over classes 1–4 |
| AC-01c | — | median of the `rate_limits` deltas, where available |

## Sample

**AC-01/AC-02:**

| Classes | Legacy runs | New runs | Total |
|---|---|---|---|
| 1–4 | 1 each | 2 each | 12 runs |

These are **indicative point estimates (low n)**. No statistical significance is claimed. Corroborate with ≥ 3 historical matched tasks per class.

**AC-04:** the sum of `ac04` over engine-executed packets.

| Source | LLM packets |
|---|---|
| W4-P01 | 8 |
| W4-P02 | 12 |
| W4-P04 | 1 |
| W4-P05 | 1 |
| **Total** | **22 (≥ 20)** |

Plus any W5-P02 corrective packets. Bootstrap packets are **excluded**. An unplanned intervention is any Operator action not listed in `contracts/operator-actions.md`.

## Other reporting rules

- Cold and warm cache are reported separately.
- Overhead is included.

## Privacy

Committed artifacts contain only counts, ids, paths, models and hashes. No message, prompt or tool text. No secrets.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P00.md -->
# W0-P00 — Isolation and bootstrap setup (Operator, OA-1)

<!-- igx:immutable begin -->
The Operator performs OA-1 in `contracts/operator-actions.md`.

## `C:\Users\igx-agent\.claude\settings.json`

```json
{
  "effortLevel": "medium",
  "modelSettings": {"claude-opus-5-5": {"effortLevel": "high"}, "claude-sonnet-5-5": {"effortLevel": "medium"}},
  "env": {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200k", "CLAUDE_CODE_SUBAGENT_MODEL": "sonnet"},
  "permissions": {
    "defaultMode": "dontAsk",
    "allow": ["Read", "Edit", "Write", "Grep", "Glob",
      "Bash(git status*)", "Bash(git diff *)", "Bash(git log *)", "Bash(git show *)", "Bash(git rev-parse *)",
      "Bash(git merge-base *)", "Bash(git fetch *)", "Bash(git add *)", "Bash(git commit *)", "Bash(git worktree list*)",
      "Bash(git branch *)", "Bash(git for-each-ref *)", "Bash(git ls-remote *)",
      "Bash(uv run pytest *)", "Bash(uv run ruff *)",
      "Bash(uv run python -m scripts.governance.agent_ops.*)", "Bash(uv run python -m scripts.governance.igx_gov *)",
      "Bash(uv run python scripts/governance/plan_digest.py *)", "Bash(uv run python scripts/governance/sync_agent_env.py *)",
      "Bash(rg *)", "Bash(hc run --bg --no-notify *)", "Bash(hc ls*)",
      "Agent(igx-researcher)", "Agent(igx-implementer)", "Agent(igx-reviewer)", "Agent(Explore)"],
    "deny": ["Bash(git push*)", "Bash(gh *)", "Bash(curl *)", "Bash(git config*)", "Bash(*--no-verify*)",
      "Bash(*INTERGRAX_SKIP_*)", "Bash(git reset --hard*)", "Bash(uv run python -c*)", "Agent(general-purpose)"]
  }
}
```

Anonymous GitHub reads go through `ao/gh_read.py` (stdlib `urllib`, public repository). Every write goes through the spool to the publisher.

## Start prompt

> Execute campaign AGENT-OPS-X autonomously per `D:\Projekty\_wt\intergrax\_agent\inbox\AGENT-OPS-X-plan-v2.1.md`.
>
> Begin with packet W0-P01. Locate its section and the `contracts/bootstrap.md` and `contracts/packets.md` sections with `rg -n "igx:file"`, then read only those sections with an offset.
>
> Follow BT-1. Stop only with `ESCALATION E<n>:`.
>
> `/goal` all W0–W3 packets implemented and `boot status` shows `engine_running`, or ESCALATION, or stop after 400 turns.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P01.md -->
# W0-P01 — Tickets, split, bootstrap tool, publisher

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W0-P01`.

## Steps

1. **Ticket number.** `NN` = max(numbers in `governance/tickets/`, `git ls-remote` branch names, and PR titles via `gh_read`) + 1.
2. **Create tickets** `governance/tickets/T-IGX-NN-{A..F}.md`, one per wave, from the README template, with these fields:
   - **Scope:** the wave's packet ids
   - **Plan digest:** output of `uv run python scripts/governance/plan_digest.py governance/campaigns/AGENT-OPS-X/source/plan-v2.1.md`
   - **Approved:** `operator @ <plan-approval ISO> (AGENT-OPS-X v2.1 plan approval; bootstrap legacy form)`
   - **Test contract:** the wave's `test_commands`
   - **Out of scope:** `forbidden_globs`
   - **Lifecycle state:** `in_progress`
3. **Copy the inbox plan** verbatim to `source/plan-v2.1.md`.
4. **`ao/split_plan.py`:**
   - split at the file markers
   - enforce the canonical format and fail-closed kinds from `contracts/authorization.md`
   - write `manifest.json`
   - must be idempotent
5. **`ao/boot.py`** with these subcommands:
   - `validate` — the mechanical checks from annex B:
     - JSON parses
     - DAG is acyclic, with no forward references
     - every `uses` is provided by a transitive dependency or is pre-existing
     - every `edit_scope` is covered by `charter.edit_globs` and disjoint from `forbidden_globs`
     - every `test_command` matches the executor's allow list
     - `waves.md` agrees with the JSON
     - canonical format holds in every section
     - every packet file is ≤ 6 KB
   - `next`, `done`, `status`, `integrate`, `publish-wave`, `run <D packet>`
   - `budget check`: a stub with an $8 allowance until W0-P04
   - `external <event>`: detects the events listed in `packets.md`
6. **`ao/gh_read.py`:** anonymous `urllib` reads of the public repository, with ETags.
7. **`ao/publisher.py`** (≤ 350 lines):
   - the App JWT, via `pyjwt` and `cryptography`
   - narrowed installation tokens
   - the typed spool operations, `selftest`, `restore-verify`, and the pause handling from `contracts/publication.md`
   - fetch-then-push from its own clone only
   - a dedup key per request
   - never prints a token
8. **Trust chain** in `ao/trust_chain/`:
   - `run_publisher.py` (≤ 80 lines): the launcher and its pause handling.
   - `update_trusted.py` (≤ 120 lines): the trusted-revision update rule from OA-2a.
   - `publisher.lock`: a hash-pinned, wheels-only lock of `pyjwt`, `cryptography` and their dependencies, produced with `uv export --no-emit-project --hashes` from the locked environment and trimmed to those packages.
   - `trust_chain.json`: the list of trust-chain files.
   - `REVIEW.md`: generated, with each file's SHA-256 and line count.
9. **Commit** with trailers `Ref: governance/tickets/T-IGX-NN-A.md` and `Packet: W0-P01`.

## Acceptance

- `boot validate` is green on the real split tree.
- A second split run produces no diff.
- The publisher tests run against a fake GitHub and spool:
  - every type outside the allowed set is rejected
  - a branch that fails the pattern is rejected
  - no git operation runs inside the agent-clone path
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P02.md -->
# W0-P02 — Isolation, capability and secret-exposure probe

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W0-P02`. Delivers `ao/probe.py`.

## Subcommands

### `isolation`

Pass/fail checks:
- `whoami` returns `igx-agent`, and that account is not an admin.
- Read access is denied to:
  - the Operator's `GitHub CLI\hosts.yml`
  - the Operator's `.claude\`
  - `C:\Users\igx-pub\`
  - `C:\Users\igx-pub\trusted\`
  - `D:\Projekty\intergrax\.git\config`
- `cmdkey /list` shows no GitHub entries.
- No `ANTHROPIC_*` or `CLAUDE_CODE_USE_*` variable is set, and there is no `apiKeyHelper`.
- The stream-json init record shows the subscription auth source. Record the key name.

### `capabilities`

Check each of the following, and name the fallback if it is missing:
- `--max-turns` (it is not shown in `--help`; fallback: `--agents` `maxTurns` plus a lease)
- `--append-system-prompt-file`
- `--session-id`
- the JSON result keys
- `--exclude-dynamic-system-prompt-sections`
- `hc` flags
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `bashOutputMaxChars`, `skillOverrides`, `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`
- what `advisorModel` costs
- whether `pyjwt` and `cryptography` import
- whether Edit of `.claude/settings.json` works under `dontAsk`. Fallback: `boot apply-settings` writes the file from the content specified in the packet.

### `overhead`

First-turn context with the default settings vs minimal settings.

### `workflow-preflight`

Parses every workflow and local composite action (`.github/workflows/*.yml`, `.github/actions/**`) with PyYAML from the locked environment, and evaluates the **effective** configuration of each job. A job **fails** if any of these hold:

1. **Token permissions.**
   - The effective `permissions` (job over workflow) are not declared explicitly, so the result would depend on the repository default.
   - It declares any write scope (`contents`, `pull-requests`, `actions`, `packages`, `statuses`, `checks`, `id-token`, and so on) on a trigger reachable from `agent-ops*` branches or PRs.
2. **OIDC.** `id-token: write` on a reachable trigger, or use of an identity-federation action (`aws-actions/configure-aws-credentials`, `google-github-actions/auth`, `azure/login`, or a custom token-exchange step).
3. **Secrets and environments.** `secrets.*` other than `GITHUB_TOKEN` on a reachable trigger, `secrets: inherit`, or an `environment` without required reviewers.
4. **Reusable workflows and actions.**
   - A `uses:` of a reusable workflow, local or remote, that does not meet items 1-3.
   - A remote reusable workflow or action that is not pinned to a full commit SHA.
   - A local composite action reached from a reachable trigger that executes PR content.
5. **Untrusted code.**
   - `pull_request_target`, `workflow_run` or `issue_comment` that checks out or executes PR-head content.
   - Expression injection: a `${{ github.event.* }}` expression or `github.head_ref` used directly in `run:` or `script:` (branch names and titles are agent-controlled).
   - `runs-on: self-hosted`.

"Reachable trigger" means `push` (any filter that can match `agent-ops/**`, or no filter), `pull_request` from the same repository, `pull_request_target`, `workflow_run`, `issue_comment`, or `workflow_dispatch` where the App can dispatch.

Writes `evidence/workflow-preflight.md` and `evidence/workflow-preflight.json`, with one row per workflow and job, its effective permissions, and a **PASS** or **FAIL** result. Any FAIL raises E5, and a patch is written to `evidence/patches/workflow-guard.md` for OA-3a.

**The publisher makes PASS mandatory.** Before the first publication and before every push batch it recomputes the preflight on the trusted clone, and publishes nothing unless the result is PASS.

### `freeze-snapshot`

Delivered here; used by W0-P07.

## Acceptance

- `isolation` passes.
- `workflow-preflight` result is recorded.
- No push happens in this packet. App push tests run in the publisher `selftest` (OA-2a), after the preflight.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P03.md -->
# W0-P03 — Baseline evidence

<!-- igx:immutable begin -->
- **Contract:** `contracts/packets.md#W0-P03`
- **Run:** `boot run W0-P03` writes `evidence/baseline.md` from annex H of `source/plan-v2.1.md`, together with the method, the cutoff and the corrections.
- **Acceptance:** every baseline needed for AC-01..AC-07 is present and cites its source.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P04.md -->
# W0-P04 — Usage analyzer and bootstrap budget check

<!-- igx:immutable begin -->
- **Contract:** `contracts/packets.md#W0-P04`
- **Deliverables:**
  - `ao/usage.py`, plus `igx_gov usage {report,import}` registered in `cli.py`. It implements `contracts/measurement.md`. Prices live in `ao/pricing.json` and are labelled as an assumption.
  - A real `boot budget check` that imports the `igx-agent` transcripts since W0-P01 into `.git/igx-ops/boot-ledger.json`. It applies the wave allocations, the W5 reserve and the margin. Bootstrap runs in a single session.
- **Tests:**
  - duplicates are counted once
  - 5m/1h split
  - synthetic rows excluded
  - privacy
  - refusal at 100 %
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P05.md -->
# W0-P05 — Enterprise impact and ACT-N11 baseline

<!-- igx:immutable begin -->
- **Contract:** `contracts/packets.md#W0-P05`
- **Executor:** read-only researcher
- **Output:** `evidence/enterprise-impact.md` (≤ 6 KB, with `path:line` references), answering:
  1. EBH-7 coverage of `edit_globs`.
  2. TRUSTED_POLICY_PIN entries and how they are re-pinned.
  3. The effect of D-02 on FRZ-AUD-10 / OPEN-BP-01.
  4. Whether FRZ-FRZ-10 or FRZ-REG-01..03 impose any obligation.
  5. **Triggers:** whether `ci-smoke` and the planned `agent-ops-*` checks run on PRs to `development`. If not → E2, with a workflow-trigger patch for OA-3.
  6. The G6 independence conditions.
- **Also delivers:**
  - `probe act-n11-baseline`, which runs the IIAC verifier read-only through `uv`
  - its output, `evidence/act_n11_baseline.json`
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P06.md -->
# W0-P06 — Benchmark and pilot selection

<!-- igx:immutable begin -->
- **Contract:** `contracts/packets.md#W0-P06`

## Output

`contracts/benchmark.md`, kind `generated`, ≤ 6 KB, containing:

- **4 A/B tasks**, one each for classes 1–4.
- **The pilot wave:** exactly **12 packets**. Each is listed with an id, a `read_scope`, an `edit_scope` and `test_commands`, in the same JSON shape as `packets.md`.
- **Historical analogs:** at least 3 per class.

## Constraints

- Everything runs on `agent-ops-bench/*` only.
- No frozen-campaign globs (taken from W0-P07). No IIAC set.
- No campaign packet.

## Validation

`probe bench-check` validates the constraints and appends the digest to `generated.lock`.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W0-P07.md -->
# W0-P07 — Frozen-campaign snapshot

<!-- igx:immutable begin -->
- **Contract:** `contracts/packets.md#W0-P07`
- **Run:** `probe freeze-snapshot`
- **Output:** `evidence/frozen-campaigns.md`, holding open PRs with head SHAs (via `gh_read`), branch tips (`git ls-remote`), the latest `Checkpoint:` lines, and a machine-readable `globs` section listing each frozen campaign's files and branches.
- **Constraint:** nothing is modified. The Operator's checkout cannot be inspected; record it as such.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P01.md -->
# W1-P01 — L0 diet

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P01`

## Steps

1. Create `agent-env/rules/intergrax-core.md` (always loaded, ≤ 4 KB). It contains:
   - compressed rules: branch safety, attribution, session artifacts, test budget, token use
   - an `igx:authority` autonomy block that points to `contracts/escalation.md`
   - the bootstrap resume rule
2. Retire the always-on rules that core now absorbs, including the duplicate DCP bullet.
3. Turn `env-vendor-qualification` into an on-demand skill. Core keeps the 2-line C10/C15 note.
4. Shrink `CLAUDE.md` to ≤ 25 lines and `AGENTS.md` to ≤ 60 lines.
5. Add the `igx-worktree` skill, including the `_agent` layout.
6. Fix the duplicate at `intergrax-iteration.md:18`.
7. Add size gates to `sync_agent_env --check`:
   - L0 ≤ 8 KB
   - `CLAUDE.md` ≤ 25 lines, `AGENTS.md` ≤ 60 lines
   - skill description ≤ 300 chars
   - skill body ≤ 120 lines, unless listed in `size_exemptions.yaml`
8. Regenerate the adapters.

## Acceptance

- Gates are green.
- The PR lists an old-rule → new-section mapping showing nothing was lost.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P02.md -->
# W1-P02 — Skill and MCP diet

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P02`

## Changes

- `skillOverrides`:
  - `name-only` or `off` for skills unused in execution
  - `user-invocable-only` for `sfx-*`
- MCP stays deferred.
- `disabledMcpjsonServers` for the servers that aren't needed.
- `AGENT_TOKEN_ECONOMY.md` (≤ 120 lines): the per-session connector checklist and how to read `/context`.

## Acceptance

The first-turn delta is measured.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P03.md -->
# W1-P03 — Typed agents

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P03`

## Agents

Canonical definitions live in `agent-env/agents/` and are generated into `.claude/agents/`. Each body is ≤ 40 lines.

| Agent | Model / effort | Tools | Max turns | Notes |
|---|---|---|---|---|
| `igx-researcher` | haiku / low | Read, Grep, Glob | 15 | — |
| `igx-implementer` | sonnet / medium | Read, Edit, Write, Grep, Glob, Bash | 60 | `isolation: worktree` |
| `igx-reviewer` | sonnet / medium (opus for M2S/M3/EC) | Read, Grep, Glob | 20 | — |
| `igx-architect` | opus / high | Read, Grep, Glob, Write | 40 | — |

## Other changes

- Deny `Agent(general-purpose)`.
- `role_registry.yaml` `subagent_role_map`, with a re-pin.
- `igx-devos-*` role skills, each ≤ 30 lines.

## Acceptance (recorded)

- A general-purpose agent is denied.
- The reviewer has no Bash.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P04.md -->
# W1-P04 — Model, effort, compaction window

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P04`

## Project `.claude/settings.json`

- `effortLevel: medium`
- `env.CLAUDE_CODE_AUTO_COMPACT_WINDOW=200k`, or `CLAUDE_CODE_DISABLE_1M_CONTEXT=1`, according to W0-P02

## Advisor

If `advisorModel` costs anything, scope it down or disable it, with evidence.

## Acceptance

Sonnet runs at `medium`, and compaction happens at ≤ 210k (`w1-checkpoint`).
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P05.md -->
# W1-P05 — Wake-up elimination

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P05`

## Core rule

- No Monitor, `list_sessions` or `git fetch` polling.
- Status lives in state (boot or engine).
- SendMessage only for DONE or ESCALATION.
- Waiting on CI is done by deterministic code.

## hc

`hc` with `--no-notify` (`notes=false` was set in OA-1).

## Acceptance

Wake-up turns are under 25 % of baseline.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P06.md -->
# W1-P06 — Memory policy and optional Operator patch

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P06`

## `igx-agent` memory

- ≤ 12 lines and ≤ 1.5 KB, pointers only.

## `evidence/operator-profile-patch.md`

- Optional (OA-6), ≤ 3 KB.
- Contains:
  - a consolidated `MEMORY.md` proposal, built from the memory index lines quoted in annex I
  - `modelSettings`
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P07.md -->
# W1-P07 — Read and Bash guards

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P07`

## Output cap

`bashOutputMaxChars: 12000`

## `igx_read_guard.py`

PreToolUse on `Read`, stdlib only, < 150 ms.

- **Deny:**
  - **Large file:** an un-ranged read of a file > 80 KB or > 1500 lines. The denial reason includes ≤ 30 heading/def line numbers.
  - **Repeat:** the same path and range, with unchanged mtime and size, in the same context epoch.
- **Context epoch:** advances on SessionStart(compact|resume) and on PostCompact.
- **Insist:** an identical retry immediately after a denial is allowed.
- **Exemption:** the `auditor` role is exempt from the repeat rule.
- **`tool-results/` files** always count as large.

## `igx_bash_guard.py`

Denies only `cat`/`type`/`Get-Content` on files > 80 KB, and `sed -n` ranges > 600 lines.

## Kill switch

`IGX_HOOKS=off` disables only these two hooks.

## Acceptance (W4)

- False positives < 5 %.
- Largest tool result ≤ 20k chars.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W1-P08.md -->
# W1-P08 — W1 checkpoint

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W1-P08`

## Run

`boot run W1-P08` runs `probe overhead` and `igx_gov usage report --since <W1 integrate>`, then writes `evidence/w1-checkpoint.md`.

## If AC-05 is unreachable

Report the decomposition and the alternatives. This is a report, not an escalation.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P01.md -->
# W2-P01 — Charter verifier and approval binding

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P01`

## Deliverables

- `ao/charter.py` and `igx_gov charter {digest,verify,covers,amend}`, implementing `contracts/authorization.md` exactly. The approval fetch goes through `gh_read` (public reviews API).
- `governance/agent-control-plane/agent_ops.json`, containing:
  - `operator_login`
  - `agent_identity`
  - `valid_until_days: 120`
  - `cache` lists

## Tests

- Every must-fail and must-pass case in `contracts/authorization.md`.
- A real-tree `verify` against the actual split.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P02.md -->
# W2-P02 — Classifier, gate, workflow patch

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P02`

## Deliverables

- `ao/classify.py`
- `risk_map.json`, which lists `contracts/**`, `packets/**`, `generated.lock` and `manifest.json` as M2/M2S, and the trust-chain files as M2S
- `ao/gate.py`
- `evidence/patches/agent-ops.yml`

## `agent-ops.yml`

- **Triggers:** `pull_request`, `pull_request_review`, and `push` to `development`. On `push` the gate reports success without checks, which registers the check sources.
- **Job `agent-ops-gate`:** base code only.
- **Job `agent-ops-tests`:** PR code, in a separate runner.
- **No secrets.**
- **Permissions:** read-only.

## Gate checks

- charter `verify` + `covers`
- paths ⊆ `edit_scope`
- declared class ≥ computed class
- approvals at head for the computed class
- IIAC set and frozen globs untouched
- `Ref:` and `Packet:` trailers present
- lifecycle state
- ACT-N11 delta
- non-chartered PRs only for legacy tickets

## Tests

- a risk-map table
- content signals
- an edit to `gate.py` → M2S
- declared < computed → fail
- a stale-SHA approval is ignored
- `sys.path` isolation

## Before OA-3

The publisher runs `gate.py` from the trusted clone as a merge precondition.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P03.md -->
# W2-P03 — `merge.py` (SHA-bound, publisher-side)

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P03`

## Deliverable

`ao/merge.py`, run inside the publisher from the trusted clone.

## Behaviour

- Applies the preconditions in `contracts/publication.md`.
- Merges with `--match-head-commit`.
- Is idempotent.

## Tests

- head moved
- check pending
- reviewer == implementer
- path outside scope
- argv
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P04.md -->
# W2-P04 — Permission settings

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P04`

## Project `.claude/settings.json`

- The W0-P00 allow and deny lists.
- `Edit` and `Write` denied for:
  - `.claude/settings*.json`
  - `.githooks/**`
  - `.github/**`
  - `governance/agent-control-plane/**`
  - `governance/campaigns/*/{REVOKED,manifest.json,approval.json}`
  - the IIAC set
- Hook registration.

## Worker settings

`ao/worker_settings/<role>.json`, used with `--permission-mode dontAsk`:

- An M2 packet gets a generated, per-packet allow list derived from its `edit_scope`.
- No `gh`, `git push`, `Agent` or `curl`.

## Acceptance

Zero prompts during engine runs.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P05.md -->
# W2-P05 — STOP-rule removal and autonomy lint

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P05`

## Rewrite

Rewrite the `read_scope` files, keeping the independent-audit wording unchanged.

Replace these phrases with an E1–E8 pointer:
- "STOP and ask"
- "wait for OK"
- "unless asked"
- "on operator request only"
- "Do not commit unless"
- "one file at a time"

## Lint

`igx_gov lint autonomy`:
- Uses `agent-env/autonomy_allowlist.yaml`. Each allowed occurrence needs a reason and an E-code.
- Excludes `agent-env/plans/**`.
- Runs in `doctor` and in the gate.

## Acceptance

- The lint is green.
- The PR contains a mapping table.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P06.md -->
# W2-P06 — Lifecycle and approval unification

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P06`

## Steps

1. Update `verify_plan_approval.py`:
   - Accept the `campaign-charter` form, only through `charter.verify` + `covers`.
   - The legacy form must have an approver listed in `plan_approvers.yaml`, and the ticket must be in `legacy_tickets.txt`. That list is frozen at activation; tickets A–F stay on it until W2-P08.
2. Make `verify_ticket_lifecycle.py` validate transitions.
3. Wire both into the gate and into `run_push_gate.py`.
4. Update `task_lifecycle.yaml` and `plan_approvers.yaml` (add `campaign-charter`).
5. Extend `dcp-commit-ref-config.yaml` to `.claude/`, `agent-env/`, `.githooks/`, `governance/` and `.github/`.
6. Remove `--force` from `ticket_lock.py`.
7. Re-pin once.
8. Update the README.

## Tests

- Extend the registry and routing tests.
- Add a caller-enumeration test.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P07.md -->
# W2-P07 — Patches, ruleset spec, GitHub probe

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P07`

## Prepare

- `evidence/patches/CODEOWNERS`, generated from the M2/M2S entries in `risk_map.json`. It owns itself and `.github/**`. A test keeps it in sync.
- `evidence/patches/agent-ops.yml`
- `governance/agent-control-plane/github/rulesets/{development,main}.json` (contents defined in `contracts/publication.md`, Ruleset bootstrap order, step 3)
- `docs/project/technical/devos/OPERATOR_ACTIONS_AGENT_OPS_X.md`, which renders OA-3

## `probe github`

Uses the publisher's `poll` and `selftest` results plus anonymous reads. It verifies:
- ruleset fields
- check sources = Actions
- strict policy
- 0 required approvals
- code owners
- stale dismissal
- no bypass
- the App cannot bypass

It then records `oa3_applied`.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W2-P08.md -->
# W2-P08 — Re-bind to the charter approval

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W2-P08` (external: `w0_pr_approved`)

## Run

1. `igx_gov charter verify AGENT-OPS-X --approval-pr <W0 PR>`, which writes the derived `approval.json`.
2. Move tickets A–F to the `campaign-charter` form.
3. Remove them from the legacy list.

## Acceptance

`verify` passes.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P01.md -->
# W3-P01 — State store

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P01`

## Deliverable

`ao/state.py` per `contracts/state.md`:
- migrations
- transition table
- epoch fencing
- backups
- snapshot
- `import-boot`

**Windows:** bounded retry on a locked database; never hold a transaction across a subprocess.

## Tests

- illegal transition
- second engine refused
- takeover increments the epoch
- old engine's write rejected
- crash consistency
- restore
- import
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P02.md -->
# W3-P02 — Scheduler

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P02`

## Deliverable

`ao/scheduler.py` reads the `packets.md` JSON only.

## Ready set

A packet is ready when all of these hold:
- its deps are `implemented`
- its externals are satisfied
- it has no lease
- its `edit_scope` is disjoint from every leased packet's `edit_scope`
- concurrency is below `min(3, headroom)`
- a budget reservation is possible

## Behaviour

- Wave transitions are automatic.
- Integration is `--no-ff` with a `Packet:` trailer.
- A merge conflict triggers a corrective attempt.
- The PR opens per `waves.md`.

## Tests

- ordering
- conflicts
- externals
- transition
- PR preconditions
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P03.md -->
# W3-P03 — Runner

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P03`

## Deliverable

`ao/runner.py`, running as `igx-agent`.

## Per attempt

1. Create a worktree.
2. Write `runs/<attempt>.launch.json`.
3. Launch:

   ```
   claude -p --session-id <uuid> --model <m> --effort <e> --max-turns <t> --max-budget-usd <reservation> --output-format json --permission-mode dontAsk --settings <worker settings> --disallowedTools Agent --append-system-prompt-file <role> "<prompt ≤ 1.5 KB>"
   ```

   Apply the W0-P02 fallbacks.
4. Environment: allowlist only. Refuse to launch if any API or provider variable or `apiKeyHelper` is present. Never `--bare`.
5. Launcher: `hc run --bg --no-notify --outlive-parent`. If `hc` is unavailable, fall back to a detached subprocess and log it.
6. Write the result to `runs/<attempt>.json`, then settle the budget.
7. Auth check: anything other than the subscription → kill, E6.
8. Rate limit → `waiting_external` until reset. No paid fallback.

## Tests

- argv
- environment scrub
- API-key refusal
- parsing
- orphan detection
- fallback
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P04.md -->
# W3-P04 — Quality pipeline and cache

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P04`

## Deliverable

`ao/quality.py`, which runs these in order:
1. commits exist
2. diff ⊆ `edit_scope`
3. `ruff`
4. `test_commands`, in a single pytest invocation
5. `igx_gov ticket --profile fast`
6. `sync --check`, when relevant
7. `lint autonomy`

Results are cached per `contracts/state.md`.

## Failure handling

- The failure digest is ≤ 1.5 KB.
- The same signature twice → E4.

## Tests

- cache key
- `never` entries are never cached
- scope check
- digest cap
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P05.md -->
# W3-P05 — Review dispatch

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P05`

## Deliverable

`ao/review.py`:
- Bundles stat + diff (≤ 60 KB, split if larger) + acceptance + quality results.
- The reviewer gets only `Read`, `Grep` and `Glob`.
- The verdict has a strict format. A malformed verdict counts as FAIL.
- Re-review covers the delta only.
- The verdict reaches the PR through a publisher `comment{kind:review_verdict}`. The record stores `worker_id`.

## Tests

- parser
- delta
- reviewer ≠ implementer
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P06.md -->
# W3-P06 — Publication pipeline and roadmap index

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P06`

## Deliverable

`ao/publish.py`, which turns wave states into spool requests:
1. push
2. `pr_upsert`, using a template with packets, `Ref`, class, quality, review, and no AI attribution
3. `poll`, with backoff ≤ 10 min
4. `merge`
5. lifecycle and status updates
6. `ao/roadmap_index.py` regenerates `ROADMAP_INDEX.md` (≤ 3 KB)

Every step is idempotent and preceded by a reconcile.

## Tests

- crash after push
- crash after merge
- 5xx backoff
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P07.md -->
# W3-P07 — Budget controller

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P07`

## Deliverable

`ao/budget.py`, implementing exactly `contracts/budget.md`.

## Tests

- **Concurrency:** 8 processes reserve simultaneously; the sum of reservations never exceeds the cap.
- Orphaned reservations stay held.
- `charge_full`.
- Rotation does not reset the budget.
- The W5 reserve is untouchable.
- 70/90/100 behaviour.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P08.md -->
# W3-P08 — Stop guard and ctx

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P08`

## Deliverables

- `scripts/hooks/igx_stop_guard.py`, per `contracts/stop-guard.md`.
- `scripts/hooks/igx_ctx.py`: SessionStart, ≤ 10 lines, and binds the session.
- Registration in the project settings.

## Tests

- every row of the contract
- third block → stalled
- corrupt state → allow
- ctx after compaction
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P09.md -->
# W3-P09 — Resume, reconcile, rebuild

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P09`

## Deliverable

`igx_gov wave run|resume|reconcile|rebuild|drill|pause`, per `contracts/state.md`.

## Resume order

1. integrity check
2. epoch
3. orphans
4. GitHub reconcile (via `poll`)
5. ledger
6. rate-limit wait
7. scheduler

## Covered interruptions

- app closed
- reboot (OA-5)
- process crash
- compaction
- model limit
- GitHub or CI down
- interrupted publication
- worktree lock (bounded retry, then E6)

## Tests

One scenario per interruption.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W3-P10.md -->
# W3-P10 — Fault-injection suite and engine takeover

<!-- igx:immutable begin -->
**Contract:** `contracts/packets.md#W3-P10`. It depends on W3-P09, which transitively includes W3-P08.

## Steps

1. `test_fault_injection.py`: the W3-P09 scenarios with a fake runner, a fake spool and a fake GitHub, run as a single pytest command.
2. Run BT-4.

## Acceptance

- The suite is green.
- `engine_lock` is held.
- A `takeover` event is recorded.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W4.md -->
# W4 — Real workflow qualification (packets W4-P01..P06)

<!-- igx:immutable begin -->
Contracts: `contracts/packets.md#W4-P01..W4-P06`. Every run happens on `agent-ops-bench/*`. Draft PRs are closed without merging (D-05). The W4 evidence PR is M0 and must merge without approval; that merge is the positive test for M0/M1.

- **P01 — A/B benchmark.** Classes 1–4: legacy n = 1, new n = 2. Results go to `evidence/benchmark.md`.
- **P02 — Pilot wave.** The 12-packet pilot wave under its own charter (OA-2).
- **P03 — Recovery drills** (`wave drill all`):
  - close the app
  - kill the engine during a spool merge
  - kill the `hc` terminals
  - stop the publisher task
  - corrupt the DB
  - a stale-lock takeover while the old engine is still alive

  Expected result for every drill: 0 lost state, 0 double publication, 0 re-allocated budget.
- **P04 — Security and isolation negatives.** Each case below must be refused, and the refusal recorded:
  - a PR that edits `gate.py` without approval
  - the head moves after review
  - a direct push to `development`
  - the App approving its own PR
  - the App posting a status
  - the App pushing a workflow
  - an API merge while the gate is red
  - declared class < computed class
  - the reviewer using Bash
  - a forged `approval.json`
  - a widened glob
  - `INTERGRAX_SKIP_*`
  - `igx-agent` reading the Operator or `igx-pub` profiles, or the trusted clone
  - a worker running with `ANTHROPIC_API_KEY`
  - a worker calling GitHub
  - an untyped spool request
  - a spool push of a non-`agent-ops` branch
  - a spool merge request with a forged dedup key
- **P05 — Quality regression.**
  - The same acceptance tests pass.
  - Findings and rework are compared.
  - No mandatory test was removed.
  - `doctor` and IIAC/FRZ are unchanged, and the ACT-N11 delta is 0.
- **P06 — KPI report.** `evidence/kpi.md` covers AC-01a/b/c through AC-09, with n, cold/warm, and overhead.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/packets/W5.md -->
# W5 — Independent audit and activation (packets W5-P01..P05)

<!-- igx:immutable begin -->
Contracts: `contracts/packets.md#W5-P01..W5-P05`.

- **P01 — G6 audit.**
  - A separate session (A-09), opus / high, read-only, with role separation.
  - Inputs: the closure matrix and the evidence.
  - Output: `evidence/g6-audit.md`.
- **P02 — Corrective packets.** Each finding becomes `packets/corrective/W5-C<NN>.md`, constrained by `corrective_rules`, generated and digest-frozen (A-10). Anything new goes to E1 or E3.
- **P03 — Delta re-audit.** Run in a new session.
- **P04 — Prepare activation** (no external precondition; this packet is inside the W5 PR):
  1. The closure matrix is computed and every pre-activation row is PASS.
  2. `activation.md` is written as a **declaration**: "Activation takes effect when this PR is merged with the Operator's approval at its head". It never claims that activation already happened.
  3. The legacy ticket list is frozen.
  4. Frozen-campaign migration drafts go to `evidence/migrations/`. The migrations themselves happen after activation, one charter each.
  5. The engine opens the W5 PR through the publisher and ends the packet. It does not wait for approval.
- **Approval and merge (Operator, OA-2):** the Operator approves the W5 PR at its head, and the publisher merges it. This is an Operator action, not a packet.
- **P05 — Activate** (external `w5_pr_merged`, detected through the publisher's `poll`):
  1. Run `charter verify` on the merged `development` tree, including the Operator approval at the PR head.
  2. `wave activate` sets the campaign `active` in the state store, unfreezes the roadmap, and writes `evidence/activation-record.md` (merge SHA, approval id, verify output).
  3. That record is published as an M0 PR that merges without approval.
  4. The first wave after activation is the health check.
<!-- igx:immutable end -->

<!-- igx:file governance/campaigns/AGENT-OPS-X/closure-matrix.md -->
# AGENT-OPS-X — pre-activation closure matrix

<!-- igx:immutable begin -->
Status is computed into `evidence/closure-status.md`.

| Row | Requirement | Evidence | Verifier |
|---|---|---|---|
| C-01 | AC-01a tokens ≥ 60 % (stretch 80 %) | kpi.md, benchmark.md | W4-P06 + G6 |
| C-02 | AC-01b estimated cost ≥ 60 % | same | same |
| C-03 | AC-01c Max limits reported | same | same |
| C-04 | AC-02 time ≥ 50 % | same | same |
| C-05 | AC-03 0 continuation questions | kpi.md | same |
| C-06 | AC-04 ≥ 95 %, n ≥ 20, engine-executed | events, kpi.md | same |
| C-07 | AC-05 / AC-06 / AC-07 | w1-checkpoint.md, kpi.md | same |
| C-08 | AC-08 zero violations | W0-P02, selftest, W4-P03/P04 | G6 |
| C-09 | AC-09 spend ≤ $800 | ledger report | G6 |
| C-10 | No parallel approval path | W2-P06 caller test | G6 |
| C-11 | IIAC set untouched; ACT-N11 delta 0 | enterprise-impact.md, gate logs | G6 |
| C-12 | Autonomy lint and `boot validate` green | outputs | G6 |
| C-13 | Recovery and fencing drills | W4-P03 | G6 |
| C-14 | M0 auto-merge without approval proven | W4 evidence PR | G6 |
| C-15 | AC-10 G6 PASS after delta re-audit | g6-audit.md | Operator (W5 PR) |
<!-- igx:immutable end -->

# Planning annex (Operator review only; not split)

## A. Corrections in this revision

| # | Issue | Fix |
|---|---|---|
| 1 | Engine and workers ran as the same user that held `app.pem` | A third account, `igx-pub`, holds the key and runs only `publisher.py` from operator-approved code, as a Scheduled Task. It never executes in the agent clone (fetch-then-push from its own clone) and accepts typed spool requests only. LLM processes have no GitHub credential. Residual risk is stated in `contracts/publication.md` §Residual. Bootstrap trust is the one-file review in OA-2a. |
| 2 | Headings sat outside the immutable blocks | Canonical format: H1, then one immutable block, then mutable blocks (charter only). Kinds: immutable, source, manifest (recomputed), generated (frozen), evidence, derived `approval.json`, revocation, activation. Corrective packets are a constrained generated kind. Checked against the real file (annex B). |
| 3 | GitHub bootstrap order | Bootstrap CODEOWNERS and ruleset; App-authored PRs approved by the Operator, never self-approved. OA-3 is a maintenance window (disable → commit → configure → enable). The final ruleset has 0 required approvals, so M0/M1 merge without approval, proven by the W4 PR (C-14). OA-7 break-glass covers deadlock. No bypass actors anywhere. |
| 4 | Packet contracts were prose only | `contracts/packets.md`: one JSON block holding deps, external, provides/uses, read/edit scope, test commands, executor, model and ac04. The scheduler and gate parse it with `json.loads`. `boot validate` checks it. |
| 5 | DAG consistency | `probe.py` moved to W0-P02 (no App push there; App tests in the publisher selftest after the preflight). W3-P09 now depends on W3-P08 and W3-P10 depends on W3-P09. Bootstrap packets have `ac04` = 0. AC-04 = 22 engine packets. All bootstrap commands are in the allow list. `gh` is denied, reads go through `gh_read`, writes through the spool. `hc run` and `git ls-remote`/`for-each-ref`/`branch` were added. |
| 6 | Security preflight | `probe workflow-preflight` runs before any App push and is re-run by the publisher before every push batch. Gate integrity is covered by the server-side table, forged reviews by the residual note (M0/M1 only), and direct API use by "no credential in LLM processes; typed spool". |

## B. Mechanical validation (run on this file with `python -I`)

Result on 2026-11-20, after fixes:

| Check | Result |
|---|---|
| Sections | 51 |
| Canonical-format violations | 0 |
| Packets in the JSON | 44 |
| Forward or unknown dependencies | 0 |
| `uses` not provided by a dependency | 0 |
| `edit_scope` outside the charter | 0 |
| `edit_scope` touching forbidden paths | 0 |
| `test_commands` not permitted | 0 |
| Packets over 6 KB | 0 |
| AC-04 count | 22 engine packets; 0 bootstrap packets counted |
| Budget allocations | sum to $800 |

**Issues found and fixed in this pass:**
- `authorization.md` quoted the marker literally.
- `W0-P06` referenced `W0-P07` ahead of its position.
- `agent-env/**` overlapped the forbidden path `agent-env/plans/**`; narrowed to explicit subdirectories.

`boot.py validate` repeats the same checks fail-closed in W0-P01.

## A2. Execution conditions applied after approval (2026-11-20)

| # | Condition | Where |
|---|---|---|
| 1 | W5-P04 no longer depends on the approval of the PR that contains it. It is split into P04 (prepare, inside the PR), the Operator's approval and merge, and P05 (activate, after `w5_pr_merged`). | `packets.md`, `packets/W5.md`, `authorization.md` (`activation` kind) |
| 2 | The whole trusted runtime chain is reviewed before OA-2a: `publisher.py`, the launcher, `update_trusted.py`, the hash-pinned wheels-only `publisher.lock`, and `trust_chain.json`. A trusted-revision update rule requires Operator approval at the PR head for every trust-chain change. | OA-2a, `packets/W0-P01.md` |
| 3 | Workflow preflight (renamed from the secrets check) now evaluates effective `GITHUB_TOKEN` permissions, OIDC, reusable workflows and untrusted-code execution. PASS is required before the first publication and before every push batch. | `packets/W0-P02.md`, `contracts/publication.md` |
| 4 | Mutable blocks are excluded from digests and from the split-equality check. A generated file plus its digest is valid only if its content rules pass when recomputed, its path is an Operator-approved M2/M2S anchor, and the append-only `generated.lock` agrees. | `contracts/authorization.md` |
| 5 | `spool\PAUSE` stops publication for OA-3 and every break-glass. Resuming requires `restore-verify` to pass. The publisher also pauses itself when it detects a missing protection. | `contracts/publication.md`, OA-3, OA-7 |

## C. Operator checklist

See `contracts/operator-actions.md`: OA-1 (13 steps), OA-2a, OA-2, OA-3, OA-3a, OA-4 to OA-7.

## D. Readiness

Ready for autonomous Sonnet execution after approval and OA-1. There are no open decisions.

## H. Baseline figures (source for W0-P03)

| Measure | Value |
|---|---|
| Total tokens | 5.621B (97.9 % cache read) |
| Estimated cost | $2,961: cache read $2,016, output $389, cache write 5m $302, cache write 1h $253 |
| Cost by agent | main 41.9 %, subagents 58.1 % |
| Cost by model | Sonnet 54.6 %, Opus 45.4 % |
| First-turn context | main 86.2k; general-purpose subagent 64.3k; Explore/Plan 38k |
| Main-session context | average 357k, max 967k; 2,293 turns above 400k |
| Compactions | 22 |
| Cache rebuilds | 175 (≈ $230) |
| Thinking | 34.4 % of output |
| Effort | `high` on all 20,784 turns |
| Repeated reads | 166 |
| Largest Read | 63k chars |
| Go-aheads | 75 of 94 messages |
| Wake-up turns | 493 (~228M tokens) |
| Subagents | 165 general-purpose (77 on Opus), ~8.6M each |
| Concentration | top 3 sessions = 65 % of tokens |

## I. Operator memory index lines (source for W1-P06)

The 24 one-line entries of the Operator's `MEMORY.md` as of 2026-11-20 are consolidated into ≤ 12 pointers. The facts to keep:
- Polish chat, English artifacts
- no AI attribution
- push upstream first
- verify the PR head before merge
- IIAC ACT-N11 product-path rule
- decision waves, then autonomous execution

Frozen-campaign authorizations become one pointer to D-05.
