# Intergrax — Application Creation Guide

**Canonical author workflow for Tier-3 application environments** (`applications/<app>/`).

Architecture canon: [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §31 · §45 · §47  
Agent roster and Nexus path: [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) Step 4E · Appendix F  
Composition engine: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md) · [`applications/USAGE.md`](../../applications/USAGE.md)

**Audience:** platform engineers, product teams, LLM coding agents.

**Success metric:** scaffold → first `UnifiedTaskRunner.run_task()` through host factory in **under one hour** — **zero** Nexus forks and **zero** business logic in `host/factory.py`.

---

## 1. Mental model (architecture §47)

Tier-3 **wires** the Harness; Tier-2 **thinks**. Pick one recipe:

| Recipe | When | Key artifacts |
|--------|------|----------------|
| **47.1 Minimal lab** | Spike / echo host | `manifest.py`, `host/factory.py`, `lab_defaults()` |
| **47.2 Product** | Shipped API (STRICT) | + `serving/`, deploy triad, `ARCHITECTURE.md` |
| **47.3 Virtual org** | UC-A7 policy simulation | `OrganizationalPolicyEnvelope`, `host/policy/rules/` |
| **47.4 Simulation** | Graph/scenario host | `ApplicationGraphSpec`, `*.pipeline` capabilities |
| **47.5 Mutating prod** | Tools that change external state | + `ReliabilityProfile`, ACP-TOK gates, HITL routes |

**Never implement in Tier-3:** agent business steps, ad-hoc `NexusLoop(...)`, vendor SDK calls, `if org ==` branches.

---

## 2. Author workflow (architecture §31)

```text
1. python -m intergrax.scaffold new-stack <slug> [--profile lab|product]
2. Declare ApplicationManifest + ApplicationEnvironmentProfile (manifest.py)
3. AgentBinding.mount() roster + typed factories (host/agent_factories.py)
4. Optional ApplicationHost for HookPoint reactions (not a cognition loop)
5. host/factory.py → build_harness_host_runtime() — mount HTTP/MCP/task routes
6. pytest applications/<app>/ tests + ApplicationRunSummary assertion
7. Prod: same profile shape; tenant-specific OrganizationalPolicyEnvelope (UC-11)
```

### Progressive disclosure (DX-0.4)

| Stage | Command | Delivers |
|-------|---------|----------|
| Minimal | `new-stack --minimal` | Harness-only factory, no Docker/MCP |
| Standard | `new-application` / `new-stack` | Docker, MCP, `BUILD_AND_DEPLOY.md`, `package.json` |
| Promote | `scaffold expand <slug>` | Upgrade minimal → standard layout |

### Factory rules (§31.3)

`host/factory.py` **MAY:** load `Settings.from_env()`, call `build_harness_host_runtime`, mount routes/middleware driven by profile.  
`host/factory.py` **MUST NOT:** implement agent steps, construct `NexusLoop` with ad-hoc kwargs, import vendor SDKs.

---

## 3. New application checklist (architecture §45)

Answer **before** shipping:

```text
 1. Product hypothesis this environment tests?
 2. app_id and deployment posture (§23.1)?
 3. Roster — AgentBinding.mount for each agent?
 4. Capability routing — explicit L1 or classifier L3?
 5. Single vs multi-agent — graph_spec or pipeline token (§23.4)?
 6. Full ApplicationEnvironmentProfile — no orphan slices?
 7. wire_application_environment() — no getattr on manifest?
 8. build_harness_host_runtime() — not ad-hoc NexusLoop?
 9. All surfaces → UnifiedTaskRunner.run_task()?
10. IdentityProfile matches auth story?
11. execution_mode=STRICT for production?
12. ObservabilityProfile + ApplicationRunSummary on task completion?
13. Business logic only in Tier-2 agents?
14. Org simulation — OrganizationalPolicyEnvelope when needed (§39)?
15. Dynamic reactions — ApplicationHost hooks (§32) vs profile-only?
16. Deploy triad (Docker, BUILD_AND_DEPLOY.md, .env.example)?
17. pytest smoke for manifest + host factory?
18. Product ARCHITECTURE.md updated — not duplicated in platform plan?
```

If any item is unanswered, **do not ship**.

---

## 4. Production readiness (architecture §46)

Product hosts must pass mandatory rows P1–P10 (manifest, harness factory, wiring, STRICT mode, deploy triad, agent-only business logic).  
Run:

```bash
uv run python scripts/check_application_production_gates.py
```

Capability-specific rows (HITL, multi-agent, budget, virtual org) apply when the host claims that capability — see architecture §46.2.

---

## 5. Platform ops commands (post APP-OPS)

| Command | Purpose |
|---------|---------|
| `intergrax doctor health-app --app <id>` | Environment health score (release review) |
| `intergrax doctor diff-app --app <id> --left … --right …` | Pre-deploy environment diff |
| `intergrax apps list` / `apps show <id>` | Application registry inventory |
| `intergrax envs list [--app <id>]` | Deployed environment registry |
| `intergrax apps sync` | Refresh `build/application_registry.json` |

Prefer registry artifacts over `applications/README.md` for ops automation.

---

## 6. Verification commands

```bash
uv run pytest applications/<pkg>/tests -q
uv run pytest tests/unit/applications/ -q
uv run python scripts/check_application_production_gates.py
python scripts/check_harness_no_getattr.py
```

---

## 7. Further reading

| Topic | Document |
|-------|----------|
| Profile field map | AGENT_CREATION_GUIDE Appendix H |
| Scenario matrix / UC-A* | architecture §35 · §44 |
| Evolution (snapshot, migrations, package) | architecture §49 |
| Ops (health, registry, capability graph) | architecture §50 |
| Domain audit | [`audit/TIER3_APPLICATION_ENVIRONMENT.md`](audit/TIER3_APPLICATION_ENVIRONMENT.md) |
