# `.agents/` — Antigravity Workspace Environment for Intergrax

This directory (together with root [`GEMINI.md`](../GEMINI.md)) configures **Google Antigravity** for the Intergrax platform, mirroring the discipline, role separation, skills, and safety guards maintained for Claude Code and Cursor without modifying either environment.

## Relationship to Repository SSOT

- **Canonical rules, skills, and plans:** [`agent-env/`](../agent-env/README.md) (shared SSOT across Claude Code, Cursor, and Antigravity).
- **Canonical role & skill control plane:** [`governance/agent-control-plane/role_registry.yaml`](../governance/agent-control-plane/role_registry.yaml) and [`skill_routing_registry.yaml`](../governance/agent-control-plane/skill_routing_registry.yaml).
- **Untouched environments:** `CLAUDE.md`, `.claude/`, `.cursor/`, `AGENTS.md`, `agent-env/`, `governance/agent-control-plane/`, and `scripts/hooks/igx_stop_guard.py` are left completely unchanged so all Claude/Cursor pins and CI drift checks stay green.

## Layout

```text
GEMINI.md                                   Always-on Antigravity root rules (L0 invariants + role/tool map)
.agents/
  README.md                                 This guide
  skills.json                               Declares `agent-env/skills` directly (zero-drift SSOT link)
  skills/
    igx-autonomy/SKILL.md                   Antigravity-adapted orchestration & escalation protocol
    igx-run-plan/SKILL.md                   Antigravity-adapted plan runner (paired with Stop hook)
    intergrax-ci-hotfix/SKILL.md            Explicit slash-invocable mode -> agent-env/rules/intergrax-ci-hotfix.md
    intergrax-hep-step/SKILL.md             Explicit slash-invocable mode -> agent-env/rules/intergrax-hep-step.md
    intergrax-independent-audit/SKILL.md    Audit protocol skill -> agent-env/rules/intergrax-independent-audit.md
    intergrax-iteration/SKILL.md            Explicit slash-invocable mode -> agent-env/rules/intergrax-iteration.md
    intergrax-micro-implement/SKILL.md      Explicit slash-invocable mode -> agent-env/rules/intergrax-micro-implement.md
  rules/
    intergrax-agent-roles.md                trigger: model_decision -> agent-env/rules/intergrax-agent-roles.md
    intergrax-env-vendor-qualification.md   trigger: model_decision -> agent-env/rules/intergrax-env-vendor-qualification.md
    intergrax-independent-audit.md          trigger: model_decision -> agent-env/rules/intergrax-independent-audit.md
    intergrax-scenario-factory.md           trigger: model_decision -> agent-env/rules/intergrax-scenario-factory.md
    intergrax-ci-hotfix.md                  trigger: manual -> agent-env/rules/intergrax-ci-hotfix.md
    intergrax-hep-step.md                   trigger: manual -> agent-env/rules/intergrax-hep-step.md
    intergrax-iteration.md                  trigger: manual -> agent-env/rules/intergrax-iteration.md
    intergrax-micro-implement.md            trigger: manual -> agent-env/rules/intergrax-micro-implement.md
  agents/
    igx-researcher.md                       Read-only codebase/docs researcher (model: flash)
    igx-architect.md                        Architect-planner for plans & ticket scopes (model: pro)
    igx-implementer.md                      Scoped implementer in isolated workspace/worktree (may_commit: true)
    igx-tester.md                           Serial Test contract & igx_gov ticket verifier (may_commit: false)
    igx-reviewer.md                         Read-only code reviewer, no shell (model: pro, never G6 PASS)
  hooks.json                                Antigravity lifecycle hooks (PreToolUse + Stop)
  hooks/
    igx_agy_pre_tool_guard.py               .env deny, large-read/dump guard, git/bypass gate, subagent guard
    igx_agy_stop_guard.py                   Plan-driven Stop hook reading .tmp/session/autonomy/active-plan
```

## Division of Responsibilities

1. **Design & Architecture (`igx-architect` + `igx-researcher`):**
   - `igx-researcher` locates symbols, code paths, and docs read-only (`model: flash`, up to 3 in parallel).
   - `igx-architect` writes bounded plans in `agent-env/plans/` or scopes `governance/tickets/T-IGX-*.md` without editing product source (`model: pro`).
   - Business scenario design uses the Scenario Factory skills (`sfx-session`, `sfx-discover`, `sfx-specify`, `sfx-resolve`, `sfx-blueprint`, `sfx-integration-setup`, `sfx-test-plan`, `sfx-present`), governed by `intergrax-scenario-factory.md` (L0/L1 drafts under `.tmp/sfx/<slug>/` only).
2. **Implementation (`igx-implementer`):**
   - Executes one approved ticket or plan step inside an isolated workspace (`Workspace: "share"`/`"branch"`) or `/igx-worktree`.
   - Edits strictly within `edit_scope`, runs listed tests serially, and commits in English with `Ref: governance/tickets/T-IGX-<NN>-<L>.md` and zero AI attribution.
3. **Verification & Testing (`igx-tester`):**
   - Runs ticket Test contract commands and `uv run python -m scripts.governance.igx_gov ticket` serially (one `uv`/`pytest` process at a time).
   - Never modifies product source or commits; prepares checkpoint output for the ticket.
4. **Code Review (`igx-reviewer`):**
   - Read-only, no shell. Verifies `edit_scope`, acceptance criteria, four-tier boundaries (`intergrax/ -> agents/ -> applications/`), and secrets/attribution rules.
   - Reports severity-ordered findings; never declares G6 or IIAC PASS.
5. **Delivery Closure & Independent Audit:**
   - PR creation (`delivery-closure`) uses `igx-devos-github-pr-workflow` only when human-authorized.
   - Formal G6 / IIAC audit (`independent-quality-auditor`) must run in a **separate new Antigravity conversation** following `intergrax-independent-audit.md`.

## Lifecycle Hooks & Kill Switch

- `.agents/hooks.json` enforces Intergrax safety invariants natively on `PreToolUse` and `Stop`.
- Set `IGX_HOOKS=off` in the environment to disable the workspace guards temporarily (same kill switch as Claude Code and Cursor).
- When using `/igx-run-plan <plan.md>`, `igx_agy_stop_guard.py` reads `.tmp/session/autonomy/active-plan` and keeps the agent progressing through open `- [ ]` steps until `PLAN COMPLETE` or `ESCALATION: <type>` (with a 3-stall safety cap).
