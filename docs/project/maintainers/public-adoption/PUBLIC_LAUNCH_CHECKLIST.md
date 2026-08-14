<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Public Launch Checklist

This checklist is for maintainers preparing external-reader sessions, reviewer requests, partner discovery, or public posts.

It does not create a release, license grant, support obligation, partnership term, production claim, certification, or compliance statement. **Checklist completion does not mean external validation is complete.**

Use it together with [README.md](../../../../README.md), [docs/project/overview/FAQ.md](../../overview/FAQ.md), [docs/project/community/COLLABORATION.md](../../community/COLLABORATION.md), [docs/project/overview/ROADMAP.md](../../overview/ROADMAP.md), [docs/project/community/PARTNERS.md](../../community/PARTNERS.md), [docs/project/overview/USE_CASES.md](../../overview/USE_CASES.md), and [docs/project/builders/EVALUATION_GUIDE.md](../../builders/EVALUATION_GUIDE.md).

See [EXTERNAL_READER_VALIDATION_PROTOCOL.md](EXTERNAL_READER_VALIDATION_PROTOCOL.md) for validation methodology and [OUTREACH_KIT.md](OUTREACH_KIT.md) for recruitment templates.

---

## Audit baseline

- Branch: `development`
- Audit base: `62c47a311a32bb3e9e530bc1f2983c39c55ec74c`
- Audit remote: `62c47a311a32bb3e9e530bc1f2983c39c55ec74c`
- Required PX-11 ancestor: `b942121d0a509d059681d6f1df55ff09d7aaf6a2`
- Accepted readiness SHA: `c050b5e6bff1b69a9534b46cab82c73ad572129e`
- External reader validation: `NOT_STARTED`

## Program readiness state

```text
PX-12:
ACCEPTED / CLOSED

PRE-PX13 completion gate:
IN_PROGRESS

PX-13:
NOT_STARTED / BLOCKED ON PRE-PX13 COMPLETION

External reader validation:
NOT_STARTED
```

## Reader journey readiness

The internal route review explicitly covered:

- [Public Documentation Map](../../community/PUBLIC_DOCUMENTATION_MAP.md)
- [root README](../../../../README.md)
- [LKW Product Tour](../product/LKW_PRODUCT_TOUR.md)
- [LKW Quick Start](../../../../applications/local_workspace_application/docs/product/QUICKSTART.md)
- [LKW Platform Proof](../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md)
- [Builder Quick Start](../../builders/BUILDER_QUICKSTART.md)
- [BUILD_WITH_INTERGRAX](../../builders/BUILD_WITH_INTERGRAX.md)
- [Evaluation Guide](../../builders/EVALUATION_GUIDE.md)
- [Architecture Overview](../../architecture/ARCHITECTURE_OVERVIEW.md)
- [Use Cases](../../overview/USE_CASES.md)
- [PROOFS](../../../../docs/project/proofs/PROOFS.md)
- [Partners](../../community/PARTNERS.md)
- [Collaboration](../../community/COLLABORATION.md)
- [LICENSE](../../../../LICENSE)
- [SECURITY](../../../../SECURITY.md)

Reviewed route outcomes:

- Product: README → Product Tour → Quick Start → Platform Proof; LKW remains primary.
- Builder: Builder Quick Start → BUILD_WITH_INTERGRAX → Evaluation Guide or owning technical route.
- Architect: README → Architecture Overview → PROOFS → Evaluation Guide or technical map.
- Buyer: README → Use Cases → PROOFS → evaluation, partner route, defer or stop; negative fit is not pushed onward.
- Partner: README → Partners → pilot brief → Collaboration and LICENSE; contact follows fit and scope preparation.
- Token Optimization: README → guide → proof/claim owners; it remains secondary and does not claim universal or production-proven savings.

No unresolved navigation finding remains after the final static pass.

---

## PX-12 internal readiness evidence

This is the single internal readiness record for PX-12. Execution results and reviewed evidence are labeled separately.

### Previous readiness snapshot / blockers

The previous readiness snapshot recorded three blockers: the `transformers`/metadata collection failure, the Windows LKW `stack_start` failure, and stale Token public-claims contract failures. All three are **CLOSED** in the final accepted readiness revalidation below. The Memory tenant regression is also **CLOSED**; an explicit tenant is required and no implicit default tenant is used.

### Final accepted readiness revalidation

| Check | Mode | Result | Evidence or limitation |
|---|---|---|---|
| Accepted readiness snapshot | pinned commit | `c050b5e6bff1b69a9534b46cab82c73ad572129e` | Independently accepted final PX-12 readiness state |
| Public local links | executed | PASS | 17 scoped documents, 421 local link targets |
| Required anchors | executed | PASS | 9 explicitly required anchors; checklist anchor is this section |
| Public local assets | executed | PASS | 18 local `src`/`srcset` and Markdown image references |
| Mermaid fences | reviewed | PASS | 16 Mermaid fences; all Markdown fences structurally complete |
| `uv sync --extra dev` | executed | PASS | Exit 0 |
| `uv run intergrax doctor` | executed | PASS | All doctor checks passed |
| Global gate | executed | PASS | `4282 passed`, `0 failed`, `0 errors`, `16181 deselected` |
| Documentation gate | executed | PASS | `161 passed`, `0 failed` |
| LKW Product Quick Start | executed | PASS | Supported Windows path; answer marker `AURORA-17`; citation/source `lkw_product_quickstart.txt`; `persisted_run_verified=true` |
| LKW certification matrix | executed | PASS | `matrix_check=PASS` |
| Token unit gate | executed | PASS | `985 passed`, `0 failed` |
| Token plugin contract | executed | PASS | 35 passed |
| Token evaluation packs | executed | PASS | Corrected explicit-file command: 78 passed |
| Token vLLM no-server gates | executed | PASS | 45 passed |
| LKW deep proof | evidence reviewed | PASS | Matrix `VALID`; Windows native and Linux Docker profiles PASS; full multi-phase proof not certified by those profiles |
| vLLM live proof | evidence reviewed | PASS | Named evidence is limited to vLLM 0.23.0, Qwen/Qwen2.5-3B-Instruct, Windows Docker Desktop/WSL2 and RTX 4080 Laptop 12 GB |
| Documentation blockers | review | NONE | Static link, anchor, asset and fence findings corrected |
| Technical blockers | review | CLOSED | Transformers/metadata collection, LKW `stack_start`, and stale Token public-claims failures resolved |

### Claims and legal boundary review

- `source-available`, active R&D, LKW Backend Product Alpha / MVP and PARTIAL remain unchanged.
- No production-ready, compliance, security, universal-savings or external-validation claim was introduced.
- Permission routes remain owned by [LICENSE](../../../../LICENSE) and [docs/project/community/COLLABORATION.md](../../community/COLLABORATION.md).
- Token Optimization remains secondary, PARTIAL, and bounded by `../../../../docs/project/proofs/PROOFS.md` and its claim guardrails.

### Internal outcome

```text
ACCEPTED / CLOSED
```

PX-12 readiness audit accepted at `c050b5e6bff1b69a9534b46cab82c73ad572129e`. This is an internal readiness acceptance only; it does not claim production readiness, commercial validation, real-user validation, customer validation, or completed external validation.

External reader validation:

```text
NOT_STARTED
```

---

## PRE-PX13 completion gate

The maintainer completion gate in
[`../../overview/PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md`](../../overview/PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md)
is `IN_PROGRESS`. PX-13 cannot begin until its product, proof, runnable
user-like verification, friction, recovery, deployment/onboarding, visual
experience, clean-room walkthrough, claim-synchronization, and
no-known-material-rewrite conditions are accepted.

## PX-13 validation-wave preparation

```text
Status:
NOT_STARTED

Blocked on:
PRE-PX13 completion

External reader validation:
NOT_STARTED
```

PX-13 owns participant cohorts, immutable participant URLs, invitation placeholders, moderator preparation, session records, and real external sessions. Wave 1 preparation is not authorized: do not pin participant URLs, add participant slots, or create recruitment records. No wave-specific item is marked complete here; no fictional session result exists.

Checklist completion does not conduct sessions, record fictional feedback, or claim external validation.

---

## Related documents

| Document | Purpose |
|----------|---------|
| [EXTERNAL_READER_VALIDATION_PROTOCOL.md](EXTERNAL_READER_VALIDATION_PROTOCOL.md) | Validation methodology, tasks, scoring, privacy and completion gates |
| [OUTREACH_KIT.md](OUTREACH_KIT.md) | Recruitment and session-request templates |
| [../../EVALUATION_GUIDE.md](../../builders/EVALUATION_GUIDE.md) | Reader-facing time-boxed evaluation paths |
| [../../README.md](../../../../README.md) | Repository overview |
| [../../COLLABORATION.md](../../community/COLLABORATION.md) | Collaboration and permission model |
| [PUBLIC_ISSUE_INDEX.md](PUBLIC_ISSUE_INDEX.md) | Curated public issue map |
| [MAINTAINER_TRIAGE_PLAYBOOK.md](MAINTAINER_TRIAGE_PLAYBOOK.md) | Public issue triage rules |
