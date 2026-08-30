# DECISION_VERIFICATION — extended architecture

**Parent hub:** [`DECISION_VERIFICATION.md`](../DECISION_VERIFICATION.md)

## 1. Pipeline contract model

The Verification Pipeline evaluates exactly one **Decision Version** per invocation and returns a **Verification Result** — optionally containing a **Challenge** that requests lifecycle revision.

## 2. Ordered / composed stages

Stages run in configured order. Platform default: **deterministic before probabilistic** when both apply.

## 3. Required vs optional verifiers

| Class | Fail-closed rule |
| ----- | ---------------- |
| Required stage | Unavailable → no synthetic pass |
| Optional stage | May be skipped when disabled — never silently substitute pass |

## 4. Stage results

Each stage emits typed sub-results aggregated into the Verification Result. Conflicting required stage outcomes fail closed or route to adjudication / `UNRESOLVED` per profile.

## 5. Challenge semantics

A Challenge signals semantic insufficiency. The pipeline **does not** mutate the candidate — Lifecycle mints a new Decision Version.

## Related satellites

| Topic | Route |
| ----- | ----- |
| Pipeline / stages | [`DECISION_VERIFICATION_pipeline_stages.md`](DECISION_VERIFICATION_pipeline_stages.md) |
| Security / independence | [`DECISION_VERIFICATION_security_independence.md`](DECISION_VERIFICATION_security_independence.md) |
| Revision / failure | [`DECISION_VERIFICATION_revision_failure_semantics.md`](DECISION_VERIFICATION_revision_failure_semantics.md) |
