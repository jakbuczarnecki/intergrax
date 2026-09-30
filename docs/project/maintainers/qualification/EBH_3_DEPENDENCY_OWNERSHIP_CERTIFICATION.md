# EBH-3 — Dependency & Ownership Certification

| Field | Value |
| ----- | ----- |
| **Program stage** | EBH-3 — Dependency & Ownership Certification |
| **START_HEAD** | `2fab207f3e850f1b339523587a19906ea1e9b292` |
| **AUDITED_HEAD** | _recorded at EBH-3 completion commit on `development`_ |
| **Branch** | `development` |
| **Prerequisites** | INT-EXTCOMP-X CLOSED; INT-CONFIG-REAL-X CLOSED; AW-7C CLOSED @ `2fab207f…` |
| **Cursor status** | **EBH-3 = READY FOR AUDIT** (not CLOSED) |
| **EBH-4** | **NOT ENTERED** |

Child remediation: **EBH-3-R1** — Integrations health composition boundary + import-cycle closure.

---

## 1. Closed-world inventory (domains inspected)

| Domain / layer | Modules / surfaces | Role in dependency graph |
| --- | --- | --- |
| Contracts (`intergrax/contracts`, domain `*/contracts`) | execution, governance, control plane, integrations, memory, RAG, tools, observability ports | Tier-0 semantic ABI; downward-only consumers |
| Applications (Tier-3) | `_shared/*_wiring.py`, product hosts | Sanctioned composition / bootstrap only |
| Agents (Tier-2) | agent factories, distribution | Consumes runtime + contracts; no applications |
| Autonomous Work | scoped adaptation orchestration, AW-7C closure runtime | Orchestration; Integrations owns adaptation scope |
| Integrations | registry, catalog, factory, providers, `_shared`, contracts | Catalog + provider selection/materialization owner |
| RAG | retrieval, graph adaptation, composition | Semantics + typed adaptation; Integrations materializes providers |
| Memory | contracts, control plane | USER truth owner; IntegrationProfile seams |
| Runtime | execution, events, sandbox, integrations categories | Execution materialization; sandbox attestation |
| Execution | `runtime.execution`, Nexus admission surfaces | Single execution semantic authority (EBH-2F-R1 recert) |
| Governance | control-plane mutation, admission ports | Single permission semantic authority (GOV-X1 CLOSED) |
| Capability Qualification | qualification subjects, AW-7C path | Qualification ≠ permission |
| Tools / Skills | providers, registry wiring | ToolRuntime consumers; no second execution owner |
| Observability / Diagnostics | export ports, diagnostics interpretation | Records/interprets; does not mint execution truth (HARNESS-W5/W6) |
| Sandbox | runtime sandbox contracts | Security attestation owner for hosted execution |
| Credentials / Configuration | INT-CONFIG realization, credential contracts | Integrations-owned realization; Governance admission |
| Provider registries / factories | `integrations/registry/*` | Single sanctioned composition root per provider category |

---

## 2. Dependency graph (current-HEAD summary)

```text
intergrax/contracts (+ domain contracts)
        ↓
Domain semantics (memory, rag, integrations/contracts, autonomous_work contracts)
        ↓
Runtime orchestration (runtime.*, execution spine)
        ↓
Integrations registry (catalog → factory → resolve_from_profile)
        ↓
Provider implementations (integrations/providers/*)
        ↑
Applications (Tier-3 wiring) — composition only, no upward semantic authority
Tools — invoke contracts / sanctioned registry seams
```

**Legal downward edges:** contracts → domain; domain → runtime orchestration; orchestration → integrations registry; registry → providers; applications → registry/contracts.

**Forbidden (audited):** contract → runtime namespace (enforced by EBH-2A/2I gate; AW-7C effect contract uses TYPE_CHECKING-only runtime sandbox reference); runtime → applications; agents → applications; consumer → unsanctioned `_impl` / cross-layer `_shared` composition.

**Remediated (EBH-3-R1):** `intergrax/tools/*` and `intergrax/applications/*` must not import `intergrax.integrations._shared.health`; sanctioned surface = `intergrax.integrations.registry.health_probes`.

---

## 3. Ownership matrix (audited concerns)

| Concern | Semantic owner | Contract owner | Composition owner | Implementation owner(s) |
| --- | --- | --- | --- | --- |
| Execution lifecycle | Runtime Execution | `intergrax/contracts/execution*` | Runtime builders / Tier-3 wiring | Nexus loop, workers |
| Governance permission | Governance | control-plane mutation contracts | Governed façades → authorization port | Governance core |
| Capability qualification | Capability Qualification | qualification subject contracts | CQ provider selection | Reference + plugin providers |
| Integration Catalog | Integrations | catalog + plugin contracts | `registry/catalog`, `plugin_register` | Catalog storage |
| Provider selection / materialization | Integrations | IntegrationProfile, factory contracts | `resolve_from_profile`, `resolve` | Per-category providers |
| Configuration realization | Integrations | INT-CONFIG contracts | ExistingCapabilityConfigurationRealizationService | Strategy plugins |
| Adaptive integration (AW-7C) | Integrations (scope/strategy) + AW (orchestration) | scoped adaptation + effect contracts | Integrations adaptation service | Reference strategies |
| Credential resolution | Integrations | credential contracts | resolver composition at Integrations edge | Provider resolvers |
| Sandbox security attestation | Sandbox (runtime) | `runtime.sandbox.contracts` | Runtime sandbox session/host wiring | Hosted/local backends |
| Integration health probing | Integrations | `HealthStatus`, `IntegrationHealthProbe` | **`registry.health_probes`** (EBH-3-R1) | `_shared.health` (internal) |
| RAG retrieval / GraphRAG | RAG | RAG contracts | RAG composition helpers | Retrieval services |
| Graph store provider | Integrations | Integration GraphStore contract | Integration factory | Provider bundles |
| Memory truth | Memory | memory contracts | MemoryControlPlane / Tier-3 wiring | Store plugins |
| Observability event/evidence | Observability | export/event ports | Harness/runtime composition | OTLP adapters |
| Diagnostics | Diagnostics | diagnostics contracts | interpret-only services | Analyzers |
| Worker lifecycle / dispatch | Runtime Execution | execution ports | runtime worker wiring | Worker pools |

**Verdict:** exactly-one semantic owner holds for audited concerns; AW-7C closure did not introduce duplicate Integration Catalog, Execution, or Qualification authorities.

---

## 4. Composition ownership matrix

| Concern | Discover | Select | Materialize | Adapt |
| --- | --- | --- | --- | --- |
| Integration providers | Catalog | IntegrationProfile + registry | `factory.resolve*` | Category adapters (RAG graph, etc.) |
| Graph store (RAG) | Integrations catalog | Profile field | Integrations factory | RAG typed adapter |
| AW-7C adaptation | Integrations scope registry | Strategy SPI | Integrations service | AW runtime orchestration |
| Health aggregation | Catalog / profile slugs | `health_check_all` / `health_check_catalog_slugs` | registry resolver + probes | Tools/apps via `health_probes` |
| Configuration realization | Catalog | Governance-approved request | Strategy plugin | N/A |
| External compatibility | Extension registry | Typed extension bundle | Compatibility service | Provider evidence adapters |

No competing composition owner identified for graph store, configuration realization, or AW-7C adaptation on audited HEAD.

---

## 5. Contract purity (sample)

| Surface | Finding | Classification |
| --- | --- | --- |
| `integrations/contracts/scoped_adapted_integration_effect_execution.py` | Runtime sandbox type reference moved to `TYPE_CHECKING` only | **Remediated** (EBH-3) |
| `integrations/contracts/*` (EBH-2I rescan) | No new unregistered public-contract → runtime imports | **PASS** (gate) |
| `integrations.registry.health_probes` | Public composition facade; delegates to internal `_shared.health` | **VALID** |

---

## 6. Reverse dependencies

| Finding | Status |
| --- | --- |
| Integrations public contract → runtime namespace (AW-7C effect ingress) | **Remediated** (TYPE_CHECKING) |
| Tools/Applications → `_shared.health` | **Remediated** (EBH-3-R1) |
| Memory/RAG → applications | **None** on audited EBH-2H/2G surfaces |
| Runtime → applications | **None** (tier boundary) |

**IN-SCOPE reverse-dependency blockers:** **0** after EBH-3-R1 + contract import fix.

---

## 7. Import cycles

| Cycle | Classification | Status |
| --- | --- | --- |
| `_shared.health` → `registry.profile` (module init) → runtime integrations → observability → events → sqlite provider → categories → observability (partial) | Package-init architectural cycle | **Remediated** — `IntegrationProfile` import deferred to `TYPE_CHECKING` only; module import acyclic |
| Lazy `_shared.__getattr__` health exports | Facade; no longer masks import failure | **PASS** (subprocess gate) |

---

## 8. Package-root / lazy-import audit

| Package | Pattern | Verdict |
| --- | --- | --- |
| `integrations._shared` | Lazy health exports via `__getattr__` | Legitimate facade; import acyclic after R1 |
| `integrations.registry.health_probes` | Explicit sanctioned API | Preferred cross-layer entry |

---

## 9. Integrations health (`intergrax.integrations._shared.health`)

| Question | Answer |
| --- | --- |
| Semantic concern | Integration instance / catalog health probing + optional circuit-breaker wrapped resolution |
| Owner | **Integrations** (semantic + composition); contract types in `integrations.contracts` |
| Cross-layer consumers | Tools `health.check_*`, Applications `integration_health_wiring` — **must use** `registry.health_probes` |
| Second health owner? | **No** — Tools expose tools; Integrations owns probe semantics |
| Cycle | Caused by eager `IntegrationProfile` import at module level — **fixed** |
| Remediation | **EBH-3-R1** — TYPE_CHECKING + `health_probes` facade + architecture gate |

Internal provider modules may continue importing `_shared.health` (VALID INTERNAL DEPENDENCY).

---

## 10. Execution / Governance / Qualification

| Authority | Status |
| --- | --- |
| Execution | **Singular** — EBH-2F-R1 + AW-7C closure gates; no second engine on AW-7C baseline |
| Governance | **Singular** — GOV-X1 CLOSED; qualification ≠ permission (AW-7C) |
| Qualification | **Singular** — Capability Qualification owns subject; Integrations owns adaptation qualification inputs |

---

## 11. Integrations / RAG / AW

| Area | Status |
| --- | --- |
| Integrations catalog/selection | **Singular** — EBH-2F/2G + INT-CONFIG + AW-7C architecture locks |
| RAG provider selection | **No RAG-local registry** — EBH-2G CLOSED evidence |
| AW-7C | **CLOSED** @ `2fab207f…`; orchestration only; no ToolRuntime bypass |

---

## 12. Tenant isolation audit

```text
tenant scope applicable: YES (local cross-domain composition seams)
canonical tenant identity: platform tenant contracts on governance/configuration/adaptation paths
verdict: N/A — WITH EVIDENCE (global TENANT-X not entered; AW-7C + INT-CONFIG local tenant proofs remain prior stage evidence)
```

EBH-3 did not widen tenant scope via dependency remediation.

---

## 13. Enterprise audit matrix

| Dimension | Verdict |
| --- | --- |
| Boundaries | PASS (post R1 + contract TYPE_CHECKING) |
| Ownership | PASS (audited matrix) |
| Contracts | PASS (EBH-2I gate + AW-7C fix) |
| Composition | PASS (health_probes sanctioned owner) |
| Typing | PASS on remediated seams |
| Pluginability | PASS (unchanged; prior stage evidence) |
| Replaceability | PASS (unchanged; prior stage evidence) |
| Authority | PASS |
| Tenant | N/A — WITH EVIDENCE |
| Regression | PASS (gates below) |

---

## 14. FRZ evidence (local contribution)

| Family | EBH-3 evidence | Global PASS delta |
| --- | --- | --- |
| FRZ-BND-* | EBH-3-R1 cross-layer health boundary; import-cycle closure | **0** |
| FRZ-OWN-* | Single Integrations health composition owner | **0** |
| FRZ-CTR-* | Public contract boundary gate green after AW-7C contract fix | **0** |
| FRZ-TEN-* | No new tenant claims | **0** |

---

## 15. Tests / gates

```bash
uv run pytest -p no:xdist \
  tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py \
  tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py \
  tests/unit/architecture/test_ebh_2i_final_rescan_gate.py \
  tests/unit/autonomous_work/test_aw_7c_closure_architecture_gates.py \
  tests/unit/autonomous_work/test_aw_7c_prerequisite_architecture_gates.py
```

Result: **PASS** (see session log `.tmp/session/ebh-3/pytest-arch2.log`).

---

## 16. Unresolved findings

| Finding | Classification |
| --- | --- |
| — | **IN-SCOPE BLOCKER = 0** |

Historical EBH-2G `_shared.health` freeze debt: **CLOSED** via EBH-3-R1.

---

## 17. Program status

```text
AW-7C = CLOSED (ledger @ 2fab207f3e850f1b339523587a19906ea1e9b292)
EBH-3 = READY FOR AUDIT
EBH-3 ≠ CLOSED
EBH-4 = NOT ENTERED
global FRZ PASS delta = 0
```

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
