# Legal agent — behavioral spec extracted from legacy (pre-AA reset)

**Purpose:** Preserve requirements before the hard reset to UAEP scaffold. Implementation follows `ARCHITECTURE.md` and Phase AA-LEG.* in the implementation plan.

## Capabilities

- `legal.review` — primary contract review entry

## Skills (Tier-0)

- `legal.contract_review` — tools: `rag.retrieve`, `websearch.query`; policy fragment `legal.contract_review.policy`

## Product profiles (deferred to Band 3 / host settings)

Legacy supported `strict_legal`, `safe`, `research`, `fast` SKUs via `LegalAgentProductProfile`. Reintroduce via `ApplicationEnvironmentProfile` and host settings, not a parallel agent config object.

## UAEP steps (target graph — implement incrementally)

1. Intake / normalize user request
2. Retrieve evidence (RAG)
3. Optional web evidence
4. Risk / policy compliance check
5. Recommendation and finalize answer

## Out of scope for scaffold baseline

- Custom `legal_execution_loop` parallel to Nexus
- Direct `LegalAgentConfig` monolith
- Live LLM E2E product proof (Band 3 — K.6)
