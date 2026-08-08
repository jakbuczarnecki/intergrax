# Partner handoff — Governed External Contractor (GEC)

**Audience:** design-partner adapter authors and Intergrax operators validating the GEC proof path.  
**Status:** GEC-0 placeholder — API sequences and fixtures arrive with GEC-7 / GEC-8.  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · Plan: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

---

## Collaboration boundary

Intergrax is **source-available** for evaluation and technical partner discovery. It is **not** open source. This handoff supports proof-path validation and integration feedback under [`COLLABORATION.md`](../../../community/COLLABORATION.md). It does **not** grant production, commercial, redistribution, or SLA rights.

Do not treat GEC-0 scaffold endpoints as a partner contract. The durable partner contract is established in later GEC phases and recorded here.

---

## What Intergrax owns vs what the partner owns

| Intergrax (governed shell) | External contractor partner |
|----------------------------|-----------------------------|
| Tier-3 public task API (when shipped) | Domain execution quality |
| Tenant / env configuration | Agent Card contents |
| Policy bundles and HITL quote acceptance | Quote commercial terms |
| Workspace deliverable boundaries | Deliverable generation |
| Trace, normalized evidence, governed receipts | Partner-native receipt products (optional) |
| Provider-neutral integration contract | Partner HTTP/A2A surface |

Intergrax does **not** reimplement the partner’s contractor agent.

---

## Target integration flow (planned)

```text
Partner client or operator
  → POST Tier-3 intake (GEC-7)
  → Nexus task
  → ExternalContractorAdapterAgent
  → External A2A contractor (partner)
  → quote → HITL accept/reject
  → status sync → deliverables → governed receipt
```

---

## Base URL (scaffold baseline)

| Environment | URL |
|-------------|-----|
| Local dev | `http://127.0.0.1:8000` |
| Docker | See [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) |

Health: `GET /health`  
Scaffold run (not the final GEC proof API): `POST /v1/governed_contractor/run`

---

## Authentication

Product host settings use the `GOVERNED_CONTRACTOR_` prefix (see `.env.example`). When bootstrap API keys are configured, send the key per generated `host/settings.py` conventions. Local unauthenticated mode is for development only — not a production claim.

---

## Mapping rules (GEC-8)

Partner-specific field maps, URLs, and credentials:

- belong in **environment configuration** and this handoff document / sample fixtures,
- must **not** be copied into `intergrax/` core modules,
- should reference the provider-neutral contracts from GEC-1 / GEC-2.

---

## Planned handoff checklist

- [ ] Quote-first request/response fixtures
- [ ] Correlation field table (`run_id` ↔ external task id ↔ quote id)
- [ ] HITL accept/reject examples
- [ ] Deliverable retrieval examples
- [ ] Receipt / trace correlation notes
- [ ] Stub vs live operator runbooks (GEC-9 / GEC-10)
- [ ] PASS matrix link (GEC-11)

---

## Contact and feedback

Use the repository collaboration channels described in [`PARTNERS.md`](../../../community/PARTNERS.md) and public adoption docs. Prefer structured proof-path feedback over unconstrained feature requests.
