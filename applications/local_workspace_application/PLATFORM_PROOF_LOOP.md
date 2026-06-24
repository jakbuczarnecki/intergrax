# Local Knowledge Workspace (LKW) — Platform Proof Loop

**Status:** active governance rule for LKW implementation  
**Parent:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Hardening addendum:** [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md)  
**Plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

---

## 1. Decision

LKW is not only a product proof. LKW is the first proof that the Intergrax platform can repeatedly create, configure, run, package, deploy, observe, and evolve agent applications.

A wave is not complete when LKW works manually but the reusable platform, scaffold, configuration, Docker, CI, or deployment implications are left behind.

---

## 2. Rule

Every non-trivial LKW step must include two acceptance layers:

1. **Product acceptance** — the LKW capability works.
2. **Platform acceptance** — the reusable lesson is propagated to the platform, scaffold, settings, build/deploy path, or CI/CD surface when applicable.

This prevents proving only a hand-built LKW application while leaving the platform unable to generate the next application correctly.

---

## 3. Platform propagation loop

For every LKW wave, run this checklist:

| Step | Question | Required action |
|------|----------|-----------------|
| 1. LKW implementation | Did the product capability change? | Implement and test the LKW behavior. |
| 2. Shared platform extraction | Is the solution generic to agent applications? | Move or expose it through `intergrax/`, `intergrax/applications/_shared/`, runtime profiles, or approved shared contracts. |
| 3. Scaffold propagation | Should future agents/apps inherit it? | Update scaffold generators, templates, generated docs, env templates, Docker templates, or tests. |
| 4. Env/settings contract | Did configuration change? | Update `.env.example`, `host/settings.py`, validation behavior, and config docs. |
| 5. Packaging contract | Did dependencies or entrypoints change? | Update `pyproject.toml`, optional dependency groups, entrypoints, Dockerfile, `.dockerignore`, or build docs. |
| 6. Deploy/CI contract | Did the run path become verifiable? | Add or update tests, CI smoke, Docker build check, image run check, or deployment runbook. |
| 7. Documentation sync | Did the architecture or plan change? | Update architecture, implementation plan, and generated scaffold documentation. |

---

## 4. Required per-wave platform checklist

Use this list before closing any LKW implementation wave:

- [ ] Does this change belong only to LKW, or should it move to shared platform code?
- [ ] Should application scaffold generate this pattern for the next product host?
- [ ] Should agent scaffold generate this contract, test, or documentation pattern?
- [ ] Does `.env.example` match `host/settings.py` and production validation?
- [ ] Does `pyproject.toml` need a dependency split or optional dependency group?
- [ ] Does Docker still build from the monorepo root with the required files copied?
- [ ] Does Docker run expose the correct host, port, env profile, and healthcheck?
- [ ] Does CI need a new application smoke test or Docker build check?
- [ ] Does the deploy/runbook still describe the real execution path?
- [ ] Does the implementation plan identify both the LKW work and the platform propagation work?

---

## 5. Where propagation must land

| Area | Target examples |
|------|-----------------|
| Shared application runtime | `intergrax/applications/_shared/` |
| Runtime/kernel/orchestration | `intergrax/runtime/` |
| Agent scaffold | `intergrax/scaffold/` agent templates and tests |
| Application scaffold | `intergrax/scaffold/new_application.py`, product app templates, generated docs |
| Docker/build templates | shared Docker template writers and app `docker/` folders |
| Env/settings | `.env.example`, `host/settings.py`, config validation docs |
| Packaging | `pyproject.toml`, optional dependencies, build docs |
| CI/CD | GitHub Actions, smoke tests, Docker build/run checks |
| Documentation | LKW architecture/plan, app creation guide, application usage docs |

---

## 6. When not to propagate

Do not update platform or scaffold when the change is truly LKW-domain-specific, for example:

- a local workspace capability name;
- a user-file workflow that does not generalize;
- a domain-specific prompt or synthesis template;
- a temporary fixture used only for LKW tests.

But if the change affects how applications configure env, expose APIs, wire agents, build Docker images, run CI, emit trace, or validate production mode, it is platform-relevant.

---

## 7. Execution implication

The LKW implementation order becomes:

```text
1. Implement the smallest LKW capability slice.
2. Identify generic platform/scaffold/deploy implications.
3. Update the reusable surface in the same PR when safe.
4. If not safe, record a blocking follow-up before moving to the next LKW wave.
5. Keep tests and docs aligned with both product and platform acceptance.
```

This is the correct proof model: LKW proves the platform by forcing the platform to absorb every reusable lesson from the product implementation.
