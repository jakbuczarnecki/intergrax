# DECISION_VERIFICATION — security and independence

**Parent hub:** [`DECISION_VERIFICATION.md`](../DECISION_VERIFICATION.md)

## Producer / verifier separation

Semantic stages must use meaningfully independent provider/model configuration from the producer — or declare **non-independent** mode explicitly in profile.

## Rubric provenance

Named rubrics resolve to versioned criteria with provenance before semantic evaluation. Missing rubric → fail closed.

## Trusted vs untrusted boundary

Judge construction isolates **trusted instructions / rubric** from **untrusted candidate content** — prompt-injection resistant posture.

## Canonical identity

Verification records bind Decision ID + Version + execution identity — no default-tenant fallbacks.

## Fail closed

Missing required stage, unavailable verifier, or unresolved rubric → no synthetic pass.
