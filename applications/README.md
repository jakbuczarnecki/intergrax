# Tier-3 applications (`applications/`)

**Role:** Isolated deployable environments — manifest, host, env, Docker. **No domain logic** (that lives in `agents/`).  
**Engine:** [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) · **Layout guide:** [`USAGE.md`](USAGE.md)

---

## Available environments

| Application | Profile | Default port | Agents mounted | Purpose |
|-------------|---------|--------------|----------------|---------|
| [`poc_template_application/`](poc_template_application/) | lab | 8095 | Echo | Canonical Tier-3 scaffold reference |
| [`lab_application/`](lab_application/) | lab | 8090 | Echo, SignoffProbe, Legal, Research, … | Universal lab + `/debug/*` trace API |
| [`legal_application/`](legal_application/) | product | 8000 | LegalAgent | Contract review API |
| [`research_application/`](research_application/) | product | 8010 | ResearchAgent, SummaryAgent | Research → summarize pipeline |
| [`local_workspace_application/`](local_workspace_application/) | product | 8020 | LocalIndexer, LocalSearch, LocalSynthesizer | **LKW** — local file index, search, synthesis |
| [`dispute_sim_application/`](dispute_sim_application/) | product | 8025 | DisputeIntake, DisputeAnalyst, DisputeStrategist, DisputeScenario | **DSW** — dispute prep and court simulation |

**Agent index:** [`agents/README.md`](../agents/README.md)

---

## Product environments (business)

| Product | Plan rows | Status |
|---------|-----------|--------|
| **Local Knowledge Workspace (LKW)** | `LKW.*` in [§6.3a](../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated) | LKW.0 Done · active LKW.1 |
| **Dispute Simulation Workspace (DSW)** | `DSW.*` in [§6.3a](../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated) | DSW.0 Done · active DSW.1 |

Platform docs in `docs/` describe **how to host** applications — product scope lives in each app's `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md`.

---

## Quick start (any host)

```bash
cp applications/<app>/.env.example applications/<app>/.env
uv run uvicorn <pkg>.host.main:app --host 127.0.0.1 --port <port>
curl -s http://127.0.0.1:<port>/health
curl -s -X POST http://127.0.0.1:<port>/v1/<short_id>/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"<capability>"}'
```

Replace `<pkg>`, `<port>`, `<short_id>`, `<capability>` from the table above.

---

## Generate a new application

```bash
python -m intergrax.scaffold new-stack my_feature --profile lab
python -m intergrax.scaffold new-application my_product --profile product --agents my_agent --port 8030
```

Readiness checklist: [`TIER3_READINESS.md`](TIER3_READINESS.md) · Author guide: [`docs/AGENT_CREATION_GUIDE.md`](../docs/AGENT_CREATION_GUIDE.md) Appendix F
