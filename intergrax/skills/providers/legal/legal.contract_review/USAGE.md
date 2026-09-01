# `legal.contract_review`

**Bundle:** `legal` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

**Baseline legal contract review** for `LegalAgent`: hybrid RAG over contract corpus plus web evidence for regulatory context. The canonical legal skill - other legal packs (`clause_compare`) depend on it via `requires_skills`.

## How it works

1. Resolves `rag.retrieve` + `websearch.query`.
2. `policy_fragment_id`: `legal.contract_review.policy` - governance fragment for tool/memory bounds (trace + capability graph; auto-merge via SK-BRIDGE planned).
3. Prompt ref: `legal.contract_review.system` - bind in Legal UAEP steps.
4. At registration, `allowed_tools` is set to the two tool ids; Tier-3 `legal_skill_profile()` enables bundle + auto-extends `tool_profile`.

## How to use

```python
from intergrax.skills.providers.legal.manifests import LEGAL_CONTRACT_REVIEW
from intergrax.applications._shared.skill_wiring import legal_skill_profile

# legal_application host
env.skill_profile = legal_skill_profile()

# agents/legal/contract.py
AgentContract(id="legal", skills=[LEGAL_CONTRACT_REVIEW], ...)
```

## What you get

| Benefit | Detail |
|---------|--------|
| **Legal SKU baseline** | Same pack on every contract-review agent |
| **Policy fragment** | Declarative governance hook for legal hosts |
| **RAG + web** | Internal corpus + external regulatory search |
| **Dependency root** | Required by `legal.clause_compare` |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Retrieve contract clauses from index |
| `websearch.query` | External legal/regulatory evidence |

## Related skills

- `legal.clause_compare` - extends with workspace diff output
- `legal.case_research` - case law + wiki search
- `rag.hybrid_qa` - generic Q&A without legal policy fragment
