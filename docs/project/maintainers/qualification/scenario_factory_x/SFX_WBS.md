# SCENARIO-FACTORY-X — WBS

**Source:** master plan R1 §10 ([`SFX-MASTER-PLAN.md`](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md)). The unit tables below are R1 verbatim plus a `ticket` column. Pre-freeze units (W0–W2) map to T-IGX-037-A..E; post-freeze units get their ticket at wave planning, after `GATE-SFX-ENTRY`.  
**Index:** [`SFX_INDEX.md`](./SFX_INDEX.md) · **Base pin:** `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`

**Columns:** ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner · Risk · Par | ticket.

**Per-unit rules:**
- One writer per file set.
- The evidence log goes to `.tmp/proof/sfx/<unit>/`.
- One `uv run pytest` per verification step.
- Every unit names its ticket at wave planning.

**Path legend for `…/` in the Files column:**
- SFX-P0-01: `…/plans/` = `governance/tickets/plans/`; `…/scenario_factory_x/` = `docs/project/maintainers/qualification/scenario_factory_x/`.
- SFX-P0-03: `qualification/scenario_factory_x/` = `docs/project/maintainers/qualification/scenario_factory_x/`.
- SFX-P1-01: `…/` = `docs/project/capabilities/plan/`.
- SFX-P1-02, P1-03, P3-*: `…/` = `docs/project/maintainers/qualification/scenario_factory_x/`.
- SFX-P2-02: `…/` = `docs/project/capabilities/plan/satellites/`; the schemas, `SFX_CANONICALIZATION.md` and `SFX_METRICS.md` all live in `docs/project/maintainers/qualification/scenario_factory_x/design/`.
- SFX-P5-02..05, P6-*, P7-*, P8-*, P9-01, P9-02 (second path), P9-03 (first path), P11-*: `…/` = `agents/scenario_factory/`.
- SFX-P10-02, P10-04..07: `…/` = `applications/scenario_studio_application/`.
- SFX-P5-04 tests: `tests/…/` = `tests/unit/agents/scenario_factory/`.

**Pre-freeze approval scope:** W0–W2 only (OD-SFX-15; STOP before SFX-P5). SFX-P4-B01/B02 were moved to W3 by R1 (§15.1: `scripts/` and `governance/agent-control-plane/` are outside OD-SFX-04).

### Pre-freeze (W0–W2; classification in §15.1)

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par | ticket |
|---|---|---|---|---|---|---|---|---|
| SFX-P0-01 | Ticket T-IGX-037-A, plan, digest, OD-SFX-01..04 records | `governance/tickets/T-IGX-037-A.md`, `…/plans/PLAN_T-IGX-037-A.md`, `…/scenario_factory_x/operator_decisions/OD-SFX-0{1..4}.md` | — | `verify_plan_approval` | ticket | Approved line by operator/delegate | architect · L · — | [T-IGX-037-A](../../../../../governance/tickets/T-IGX-037-A.md) |
| SFX-P0-02 | Roadmap A1–A10 | roadmap | P0-01 | `roadmap_replay_x` | log | no new failures vs `eadb1c21` | architect · M · — | [T-IGX-037-A](../../../../../governance/tickets/T-IGX-037-A.md) |
| SFX-P0-03 | SSOT plan + capability pair + dossier skeleton | `governance/tickets/plans/SFX-MASTER-PLAN.md`; `docs/project/capabilities/{architecture,plan}/SCENARIO_FACTORY.md`; `capabilities/README.md`; `qualification/scenario_factory_x/{SFX_INDEX,SFX_WBS,SFX_GAP_REGISTER,SFX_SHA_LADDER}.md`, `SFX_REQUIREMENTS_TRACEABILITY.json` | P0-01 | `tests/unit/governance` | log | links valid | architect · L · ∥P0-02 | [T-IGX-037-A](../../../../../governance/tickets/T-IGX-037-A.md) |
| SFX-P1-01 | Competitive evidence matrix | `…/SFX_COMPETITIVE_EVIDENCE_MATRIX.md` | W0 | citation check (each cell has URL+date or UNKNOWN) | matrix | 100% cited/UNKNOWN | research ×3 lanes · M | [T-IGX-037-B](../../../../../governance/tickets/T-IGX-037-B.md) |
| SFX-P1-02 | IDE + MCP client facts | `…/SFX_IDE_TOOLING_FACTS.md` | W0 | — | cited | resolves GAP-07/ME | research · L · ∥ | [T-IGX-037-B](../../../../../governance/tickets/T-IGX-037-B.md) |
| SFX-P1-03 | Inventory re-pin incl. CW qualification status | `…/SFX_CAPABILITY_INVENTORY.json` | W0 | JSON schema check | inventory | every CAP classified | architect · M · ∥ | [T-IGX-037-B](../../../../../governance/tickets/T-IGX-037-B.md) |
| SFX-P2-01 | ADR-SFX-001..006 + ownership review (ToolRegistry, ECAP §16) | `docs/project/technical/adr/entries/<date>/ADR-SFX-00{1..6}.md` | P1 | `check_harness_adr.py` | ADRs | independent review (reviewer session) | architect · M | [T-IGX-037-C](../../../../../governance/tickets/T-IGX-037-C.md) |
| SFX-P2-02 | Contract drafts as JSON Schema (docs) + canonicalization spec + metric definitions G-1..5 | `…/scenario_factory/design/schemas/*.schema.json`, `SFX_CANONICALIZATION.md`, `SFX_METRICS.md` | P2-01 | JSON schema lint | files | review | architect · M · ∥P3 | [T-IGX-037-C](../../../../../governance/tickets/T-IGX-037-C.md) |
| SFX-P3-01 | UX journeys, IA, static prototype (Artifact, outside the repo product paths), usability protocol, round 0 | `…/SFX_UX_DESIGN.md` | P2-01 | round 0 (5 users) | report | TTFS baseline | ux · M · ∥P2-02 | [T-IGX-037-D](../../../../../governance/tickets/T-IGX-037-D.md) |
| SFX-P3-02 | GEN set authored and sealed by an independent session | `…/SFX_GEN_SET_SEAL.md` (digest only) | P2-02 | digest verify | seal record | sealed before W3 | independent · L | [T-IGX-037-D](../../../../../governance/tickets/T-IGX-037-D.md) |
| SFX-P4-A01 | IDE v0 declarative: rule `intergrax-scenario-factory` (`loading: auto`), skills `sfx-session`, `sfx-discover`, `sfx-specify`, `sfx-resolve`, `sfx-blueprint`, `sfx-integration-setup`, `sfx-test-plan`, `sfx-present`; L1 output to `.tmp/sfx/<slug>/` against the P2-02 schemas | `agent-env/rules/intergrax-scenario-factory.md`, `agent-env/skills/sfx-*/SKILL.md` + regenerated adapters `.claude/skills/*`, `.cursor/{rules,skills}/*` (no `CLAUDE.md` block change because the rule is `auto`) | P2-02 | `test_agent_env_sync.py`, `check_cursor_token_setup.py` | logs | `--check` clean; operator walkthrough on REF-1 text | implementer · L | [T-IGX-037-E](../../../../../governance/tickets/T-IGX-037-E.md) |
| SFX-P4-A02 | Agent persona briefs (declarative, consumed by skills as subagent briefs; not synced) | `agent-env/agents/sfx-{discovery-analyst,scenario-architect,capability-resolver,integration-advisor,test-designer,critic}.md` | P4-A01 | `--check` unaffected (directory not globbed) | log | personas read-only except architect drafts | implementer · L · ∥ | [T-IGX-037-E](../../../../../governance/tickets/T-IGX-037-E.md) |

### Gate: GATE-SFX-ENTRY

All of the following must hold:
- ARCH-FREEZE is CLOSED.
- ADR-SFX-001..006 are Accepted (OD-SFX-06).
- OD-SFX-05/09/14 are signed. OD-SFX-13 is not an entry condition; it gates admission only.
- The inventory is re-pinned at `FREEZE_CERTIFICATION_SHA`.
- The GEN set is sealed.

### Post-freeze waves

Every wave ends with **REBIND-Wn**: ECAP A1–A8 delta audit, D1–D6 classification (default C), the IIAC ACT-N11 audited re-bind, and an independent reviewer session.

**W3 — foundations (P5)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par | ticket |
|---|---|---|---|---|---|---|---|---|
| SFX-P4-B01 | Extend `sync_agent_env.py` to generate agent definitions (`.claude/agents/`, Cursor per P1-02) | `scripts/governance/sync_agent_env.py`, `tests/unit/governance/test_agent_env_sync.py` | GATE | sync tests | log | `--check` clean | implementer · L · ∥ | TBD (post-freeze) |
| SFX-P4-B02 | role_registry `subagent_role_map` entries for sfx agents + pinned test update | `governance/agent-control-plane/role_registry.yaml`, `tests/unit/governance/test_agent_role_routing.py` | P4-B01 | role routing + DCP registry tests | log | 7 roles unchanged | implementer · L | TBD (post-freeze) |
| SFX-P5-01 | Tier-2 package via `intergrax.scaffold new-agent` | `agents/scenario_factory/{__init__.py,pyproject.toml,README.md}` | GATE | `test_tier_dependency_boundaries` | log | boundaries green | implementer · L | TBD (post-freeze) |
| SFX-P5-02 | Contracts CON-01..07 | `agents/scenario_factory/contracts/{problem,discovery,specification,resolution,blueprint,integration,testing}.py` | P5-01 | `tests/unit/agents/scenario_factory/contracts/` round-trip | log | 100% schema round-trip | implementer · M | TBD (post-freeze) |
| SFX-P5-03 | Contracts CON-08..17 | `…/contracts/{handoff,reporting,catalog,session,lifecycle,approval,job,qualification,admission}.py` | P5-01 | same | log | same | implementer · M · ∥P5-02 | TBD (post-freeze) |
| SFX-P5-04 | Canonicalizer (RFC 8785) + schema export + drift test | `…/canonical.py`, `…/schemas/*.schema.json`, `tests/…/test_schema_drift.py`, `test_canonical.py` | P5-02/03 | canonical vectors | log | deterministic digests | implementer · M | TBD (post-freeze) |
| SFX-P5-05 | Persistence adapters over CW, Session and LR (§7.5) incl. lifecycle-record CAS | `…/persistence/{work_items,artifacts,lifecycle,decisions,jobs}.py` | P5-04 | in-memory + SQLite backends; tenant-crossover deny | log | append-only + CAS + tenant isolation | implementer · H | TBD (post-freeze) |
| SFX-P5-06 | Tier-3 host via `new-application`; root workspace member; `.gitignore` `node_modules/`; `web/` excluded from package discovery | `applications/scenario_studio_application/{host/factory.py,serving/fastapi_router.py,manifest.py,package.json,pyproject.toml}`, root `pyproject.toml`, `.gitignore` | P5-01 | host build + `check_application_package` | log | host boots LAB | implementer · M · ∥P5-02 | TBD (post-freeze) |
| SFX-P5-07 | Service layer + CLI v0 (validate/show/canonicalize; offline L0/L1) | `agents/scenario_factory/service/*.py`, `applications/scenario_studio_application/cli/__main__.py` | P5-04, P5-06 | CLI tests | log | PC-2 for CLI | implementer · M | TBD (post-freeze) |

**W4 — discovery, design, engineering handoff (P6–P8)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par | ticket |
|---|---|---|---|---|---|---|---|---|
| SFX-P6-01 | Discovery agent (AgentContract **lifecycle STAGING explicit**), LAB via NexusLoop | `agents/scenario_factory/discovery/{agent.py,steps/,prompts/}` | W3 | scripted-interview replay fixtures `tests/fixtures/scenario_factory/discovery/` | eval report | completeness ≥ OD-SFX-07 target | implementer · H · lane A | TBD (post-freeze) |
| SFX-P6-02 | Spec author (LLM) + deterministic validator | `…/specification/` | P6-01 | validator unit + replay | log | invalid specs rejected with pointer | implementer · M · lane A | TBD (post-freeze) |
| SFX-P6-03 | Session resume (SessionManager + `SfxSessionState` artifact) | `…/service/session.py` | P5-05 | interruption/resume tests | log | resume in either door | implementer · M · lane A | TBD (post-freeze) |
| SFX-P7-01 | Capability resolver (catalog + UCA gap classes), deterministic | `…/resolution/` | W3 | golden tests | log | REF-1 golden | implementer · M · lane B | TBD (post-freeze) |
| SFX-P7-02 | Blueprint + profile composer (L2 profile: LAB, fixture adapters, egress deny) | `…/blueprint/` | P7-01 | golden blueprints | log | deterministic digest | implementer · M · lane B | TBD (post-freeze) |
| SFX-P7-03 | Integration setup assistant (secret refs only) | `…/integration/` | P7-02 | no-secret-value scan | log | 0 secret values | implementer · M · lane B | TBD (post-freeze) |
| SFX-P7-04 | Blueprint versioning + semantic diff | `…/blueprint/diff.py` | P7-02 | diff tests | log | stable diff | implementer · L · lane B | TBD (post-freeze) |
| SFX-P8-01 | `EngineeringHandoffPort` + `DETERMINISTIC_SCAFFOLD` adapter | `…/handoff/{port.py,scaffold_adapter.py}` | P7-02 | determinism (same blueprint → same tree digest) + `scenario_architecture_conformance` | log | conformant skeleton | implementer · M | TBD (post-freeze) |
| SFX-P8-02 | Engineering job lifecycle (Assignment, ExecutionLink, LR checkpoint, DLQ) | `…/handoff/jobs.py` | P8-01, P5-05 | retry/DLQ/idempotency | log | no duplicate job effects | implementer · M | TBD (post-freeze) |
| SFX-P8-03 | `AGENTIC_ENGINEERING` adapter: intent draft, human submit, SEF record tracking, bundle verification (HO-1..7); flag-gated | `…/handoff/sef_adapter.py` | P8-02 + **GATE-SEF-CONSUME** | contract tests vs SEF schemas; HO tests | log | HO-1..7 green on SEF fixtures | implementer · H | TBD (post-freeze) |

**W5 — qualification and front doors (P9, P10)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par | ticket |
|---|---|---|---|---|---|---|---|---|
| SFX-P9-01 | Test planner + runner adapter (EvalRunner, proof runner, conformance) | `…/testing/` | W4 | unit | log | REF-1 plan runs L2 | tester+impl · M · lane C | TBD (post-freeze) |
| SFX-P9-02 | Generated-code no-self-escalation gates (AST; reuse `agent_lifecycle_bypass_ast.py`) | `tests/qualification/scenario_factory_x/generated_code/test_no_self_escalation.py`, `…/qualification/escalation_rules.py` | P8-01 | mutation tests (each forbidden pattern injected → rejected) | log | 100% patterns rejected | tester · H · lane C | TBD (post-freeze) |
| SFX-P9-03 | SAST + dependency gates for generated packages and web (semgrep / ruff-S / pip-audit / npm audit per OD-SFX-14); dependency allowlist | `…/qualification/supply_chain.py`, `tests/qualification/scenario_factory_x/supply_chain/` | P8-01 | seeded-vuln fixtures | log | seeded findings detected; clean REF passes | tester · M · lane C | TBD (post-freeze) |
| SFX-P9-04 | Workspace + network qualification: scaffold writes confined; L2 egress deny; agentic path asserts SEF lease/egress evidence | `tests/qualification/scenario_factory_x/isolation/` | P8-01/03 | path-escape, egress, lease-absent tests | log | all fail closed | tester · H · lane C | TBD (post-freeze) |
| SFX-P9-05 | Adversarial suite (injection in discovery/uploads/RAG, tenant crossover, secret leakage, budget, retry) | `tests/qualification/scenario_factory_x/adversarial/` | P9-01 | suite | log | all fail closed | tester · H | TBD (post-freeze) |
| SFX-P10-01 | Studio API v1 (JWT bearer, explicit CORS, OpenAPI export) | `applications/scenario_studio_application/serving/{api_v1.py,openapi_export.py}` | W4 | API + auth tests | log | PC-2/3 for API | implementer · M · lane D | TBD (post-freeze) |
| SFX-P10-02 | MCP surface, explicit allowlist, no-widening test | `…/mcp/server.py` | P10-01 | MCP tests (streamable HTTP) | log | allowlist == SFX op set | implementer · M · lane D | TBD (post-freeze) |
| SFX-P10-03 | Parity suite PC-1..6 | `tests/qualification/scenario_factory_x/parity/` | P10-01/02, P5-07 | parity | log | 100% PC tests | tester · M | TBD (post-freeze) |
| SFX-P10-04 | Web foundation (Vite+React+TS, lint, typecheck, design system, generated client) + CI workflow | `…/web/{package.json,vite.config.ts,src/…}`, `.github/workflows/scenario-studio-web.yml` | P10-01, OD-SFX-05 | unit + build + npm audit | CI run | green, path-filtered | implementer(fe) · H · lane E | TBD (post-freeze) |
| SFX-P10-05 | Screens Describe→Configure | `…/web/src/features/{describe,discover,understand,design,configure}/` | P10-04 | unit | log | flows on REF-1 | fe · M · lane E | TBD (post-freeze) |
| SFX-P10-06 | Screens Approve→Improve + approval inbox + version history + e2e (Playwright) | `…/web/src/features/{approve,build,test,run,present,improve}/`, `…/web/e2e/` | P10-05 | e2e | log | full lifecycle walkable | fe · H · lane E | TBD (post-freeze) |
| SFX-P10-07 | SPA hosting (dev proxy + build mount in Tier-3 host) | `…/host/spa_mount.py` | P10-04 | host tests | log | no `fastapi_core` change | implementer · L | TBD (post-freeze) |
| SFX-P10-08 | IDE toolkit v1 (skills call CLI/MCP) | `agent-env/skills/sfx-*` + adapters | P10-02, P4-B01 | sync + parity | log | IDE ≡ Studio artifacts | implementer · M | TBD (post-freeze) |

**W6 — reporting, catalog, admission dossier (P11)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par | ticket |
|---|---|---|---|---|---|---|---|---|
| SFX-P11-01 | Run reporter (evidence read port) | `…/reporting/run_report.py` | W5 | golden reports | log | facts traceable to evidence | implementer · M · ∥ | TBD (post-freeze) |
| SFX-P11-02 | ROI / value model (inputs declared, no fabricated baselines) | `…/reporting/value_model.py` | P11-01 | unit | log | every number has a source ref | implementer · M | TBD (post-freeze) |
| SFX-P11-03 | Catalog sidecar + reuse/clone | `…/catalog/` | W5 | unit + "manifest unchanged" gate | log | no `ApplicationManifest` diff | implementer · M · ∥ | TBD (post-freeze) |
| SFX-P11-04 | Admission dossier export (§14.3) | `…/qualification/admission_dossier.py` | P9-*, P11-01 | dossier completeness | log | dossier ≠ admission | implementer · M | TBD (post-freeze) |

**W7 — pilots (P12)**

| ID | Description | Deps | Evidence | Acceptance | ticket |
|---|---|---|---|---|---|
| SFX-P12-01..03 | REF-1/2/3 end to end: deterministic baseline, then `AGENTIC_ENGINEERING` | W6 + GATE-SEF-CONSUME | evidence bundles @ SHA | §12 acceptance | TBD (post-freeze) |
| SFX-P12-04 | GEN set reveal, run, metrics G-1..5 (independent evaluator) | P12-01 | GEN report | OD-SFX-07 targets | TBD (post-freeze) |
| SFX-P12-05 | Usability rounds 1–3 + refinement backlog burn-down | P10-06 | reports | 0 critical defects | TBD (post-freeze) |
| SFX-P12-06 | Performance measurement §13 | P12-01..03 | metrics @ SHA | recorded with artifacts | TBD (post-freeze) |

**W8 — closure (P13)**

| ID | Description | Acceptance | ticket |
|---|---|---|---|
| SFX-P13-01 | Evidence publication + SHA ladder | complete ladder | TBD (post-freeze) |
| SFX-P13-02 | ENT-AUDIT CHANGED-SINCE + "manifest unchanged" + no-new-authority checks | 0 new violations | TBD (post-freeze) |
| SFX-P13-03 | IIAC independent **product** certification (separate session) | `SCENARIO-FACTORY-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` | TBD (post-freeze) |
| SFX-P13-04 | Operator closure OD-SFX-11 + roadmap row update | signed | TBD (post-freeze) |

**Separation of duties per unit:**
- investigation: architect;
- implementation: implementer;
- review: reviewer, in a separate session;
- per-scenario sandbox certification: an independent evaluator session;
- product certification: independent-quality-auditor;
- acceptance: operator.
