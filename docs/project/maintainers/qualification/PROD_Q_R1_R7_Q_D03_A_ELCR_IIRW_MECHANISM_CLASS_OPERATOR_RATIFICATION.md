# D03-A — ELCR-IIRW mechanism class — operator ratification OD-D03-A-01

| Field | Value |
|---|---|
| Status | **OD-D03-A-01 RATIFIED — MECHANISM CLASS ONLY** |
| Date | 2026-10-18 |
| Exact baseline | `7cc63a44fb5c31b96815a555614dcea07d618bc1` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — OPEN/BLOCKED |
| Source proposal | [D03-A trusted anchor backend & trust domain proposal](PROD_Q_R1_R7_Q_D03_A_TRUSTED_ANCHOR_BACKEND_TRUST_DOMAIN_PROPOSAL.md) |
| Production delta | **0** |

## Operator decision — bounded authority

Approve **ELCR-IIRW**, an External Linearizable Continuity Register with an Independent Immutable Recovery Witness, as the **preferred target mechanism class** for Kubernetes multi-instance history continuity (Decision-03, gate D03-A). The primary is a linearizable transactional current-head/CAS authority; the independently administered witness preserves a rollback-resistant monotonic highest sealed revision. Together they form a **single governance continuity mechanism**, not a second provider-invocation history registry.

This is approval of **OD-D03-A-01 only**. The source proposal remains a design proposal for its unratified operator selections, and its conceptual record fields/transition sequences are not executable implementations or proven vendor guarantees.

## Guardrails

1. **No service selection or infrastructure authorization:** managed primary/witness providers, cloud accounts, projects, regions, namespaces, IAM identities/permissions, credentials, budgets, backup, retention and recovery arrangements are **NOT SELECTED/NOT RATIFIED**. Operator decisions **OD-D03-A-02 through OD-D03-A-10 remain OPEN**.
2. **Independent trust domains:** witness must be outside PIS/GCA, Kubernetes/etcd and primary-register rollback/administration scope; immutable storage alone or Kubernetes Lease/ConfigMap/CRD is insufficient without a strongly consistent high-watermark. A single administrator capable of silent rollback of both domains violates the approved design.
3. **Continuity safety:** expected-version CAS, witness sealing and crash recovery must fail closed, including prepared/sealed/committed uncertainty, primary/witness mismatch, stale fence, partial/complete database restore, and disaster recovery; no cached affirmative proof or insecure allow mode.
4. **One semantic owner each:** governance lifecycle owns compatibility provenance; PIS owns invocation intents/outcomes; GCA orchestrator owns GER/F02; existing `GovernedProviderInvocationDispatchGate` remains the only physical provider execution gate, checking `create_work` only; accept/cancel retain separate semantics.
5. **Kubernetes prerequisites not waived:** D03-B one globally coordinated first-writer PIS/dispatch authority and D03-C PIS↔GCA↔anchor continuity/TOCTOU protocol are mandatory independent gates. This decision makes no exactly-once provider-effect claim.
6. **No implementation authorization:** no new service, store, database, Kubernetes resource, runtime configuration, migration, typed port or production modification; no test PASS/FRZ promotion solely from the ratification.

## Next mandatory task

**D03-A / OD-D03-A-02–08: concrete infrastructure and trust-domain qualification proposal.** Propose one actual available managed primary-register service plus one independent witness service with verified CAS/linearizability, highest-sealed-head consistency, privilege separation, DR/restore behavior, regional topology and deployment IAM; specify operator choices and evidence gaps. Vendor documentation or code references are supporting evidence, **not proof of deployment or activation**. Seek explicit operator ratification before implementing or provisioning anything.

If no candidate meets the chosen failure/trust assumptions, STOP — ARCHITECTURE DECISION REQUIRED — ANCHOR BACKEND & TRUST DOMAIN. No alternate class may silently replace ELCR-IIRW.

## Closure

D03-A **MECHANISM CLASS RATIFIED, BACKEND NOT SELECTED, NOT IMPLEMENTED/QUALIFIED**. `IMPL-01`, `R7-Q`, `PROD-Q`, all `FRZ-*` remain OPEN/BLOCKED. Exact-SHA independent audit required before any later roadmap or FRZ acceptance.
