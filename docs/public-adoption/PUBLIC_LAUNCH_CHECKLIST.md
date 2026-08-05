<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Public Launch Checklist

This checklist is for maintainers preparing external-reader sessions, reviewer requests, partner discovery, or public posts.

It does not create a release, license grant, support obligation, partnership term, production claim, certification, or compliance statement. **Checklist completion does not mean external validation is complete.**

Use it together with [README.md](../../README.md), [FAQ.md](../../FAQ.md), [COLLABORATION.md](../../COLLABORATION.md), [ROADMAP.md](../../ROADMAP.md), [PARTNERS.md](../../PARTNERS.md), [USE_CASES.md](../../USE_CASES.md), and [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md).

See [EXTERNAL_READER_VALIDATION_PROTOCOL.md](EXTERNAL_READER_VALIDATION_PROTOCOL.md) for validation methodology and [OUTREACH_KIT.md](OUTREACH_KIT.md) for recruitment templates.

---

## Audit baseline

- Branch: `development`
- Audit base: `62c47a311a32bb3e9e530bc1f2983c39c55ec74c`
- Audit remote: `62c47a311a32bb3e9e530bc1f2983c39c55ec74c`
- Required PX-11 ancestor: `b942121d0a509d059681d6f1df55ff09d7aaf6a2`
- External reader validation: `NOT_STARTED`

## Reader journey readiness

The internal route review explicitly covered:

- [Public Documentation Map](../../docs/PUBLIC_DOCUMENTATION_MAP.md)
- [root README](../../README.md)
- [LKW Product Tour](../../LKW_PRODUCT_TOUR.md)
- [LKW Quick Start](../../applications/local_workspace_application/docs/QUICKSTART.md)
- [LKW Platform Proof](LKW_PLATFORM_PROOF.md)
- [Builder Quick Start](../../BUILDER_QUICKSTART.md)
- [BUILD_WITH_INTERGRAX](../../BUILD_WITH_INTERGRAX.md)
- [Evaluation Guide](../../EVALUATION_GUIDE.md)
- [Architecture Overview](../../ARCHITECTURE_OVERVIEW.md)
- [Use Cases](../../USE_CASES.md)
- [PROOFS](../../PROOFS.md)
- [Partners](../../PARTNERS.md)
- [Collaboration](../../COLLABORATION.md)
- [LICENSE](../../LICENSE)
- [SECURITY](../../SECURITY.md)

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

| Check | Mode | Result | Evidence or limitation |
|---|---|---|---|
| Audit base | pinned commit | `62c47a311a32bb3e9e530bc1f2983c39c55ec74c` | Exact `$AuditBase`; equal to `origin/development` after concurrency recheck |
| Public local links | executed | PASS | 17 scoped documents, 421 local link targets |
| Required anchors | executed | PASS | 9 explicitly required anchors; checklist anchor is this section |
| Public local assets | executed | PASS | 18 local `src`/`srcset` and Markdown image references |
| Mermaid fences | reviewed | PASS | 16 Mermaid fences; all Markdown fences structurally complete |
| `uv sync --extra dev` | executed | PASS | Exit 0 |
| `uv run intergrax doctor` | executed | PASS | All doctor checks passed |
| `uv run pytest -m gate -q` | executed | FAIL | 150 collection errors; `TypeError: 'NoneType' object is not subscriptable` in the `transformers`/metadata import chain |
| LKW Product Quick Start | executed | FAIL | Windows marker `lkw_quickstart_result=FAIL`; `failed_stage=stack_start`; no answer marker reached |
| LKW certification matrix | executed | PASS | `--check` returned `matrix_check=PASS` |
| Token unit gate | executed | FAIL | 983 passed, 424 deselected, 2 existing public-claims contract tests failed because they require removed phase/status wording |
| Token plugin contract | executed | PASS | 35 passed |
| Token evaluation packs | executed | PASS | Corrected explicit-file command: 78 passed |
| Token vLLM no-server gates | executed | PASS | 45 passed |
| LKW deep proof | evidence reviewed | PASS | Matrix `VALID`; Windows native and Linux Docker profiles PASS; full multi-phase proof not certified by those profiles |
| vLLM live proof | evidence reviewed | PASS | Named evidence is limited to vLLM 0.23.0, Qwen/Qwen2.5-3B-Instruct, Windows Docker Desktop/WSL2 and RTX 4080 Laptop 12 GB |
| Documentation blockers | review | NONE | Static link, anchor, asset and fence findings corrected |
| Technical blockers | review | LISTED | Global gate collection, LKW stack-start, and Token public-claims contract failures remain unresolved |

### Claims and legal boundary review

- `source-available`, active R&D, LKW Backend Product Alpha / MVP and PARTIAL remain unchanged.
- No production-ready, compliance, security, universal-savings or external-validation claim was introduced.
- Permission routes remain owned by [LICENSE](../../LICENSE) and [COLLABORATION.md](../../COLLABORATION.md).
- Token Optimization remains secondary, PARTIAL, and bounded by `PROOFS.md` and its claim guardrails.

### Internal outcome

```text
CHANGES_REQUIRED_TECHNICAL
```

Technical blockers:

- Global gate collection: owner is the shared test/dependency environment and RAG embedding import chain; the documented command is correct.
- LKW Product Quick Start: owner is the local LKW stack startup/infrastructure path; the documented command is correct and failed at `stack_start`.
- Token unit gate: owner is the existing public-claims contract test; the test expects removed roadmap-phase wording, and PX-12 does not change tests or restore that mirror.

External reader validation:

```text
NOT_STARTED
```

---

## PX-13 validation-wave preparation

```text
Status:
BLOCKED_ON_PX_12_ACCEPTANCE

External reader validation:
NOT_STARTED
```

PX-13 owns participant cohorts, immutable participant URLs, invitation placeholders, moderator preparation, session records, and real external sessions. No wave-specific item is marked complete here; no fictional session result exists.

Checklist completion does not conduct sessions, record fictional feedback, or claim external validation.

---

## Related documents

| Document | Purpose |
|----------|---------|
| [EXTERNAL_READER_VALIDATION_PROTOCOL.md](EXTERNAL_READER_VALIDATION_PROTOCOL.md) | Validation methodology, tasks, scoring, privacy and completion gates |
| [OUTREACH_KIT.md](OUTREACH_KIT.md) | Recruitment and session-request templates |
| [../../EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md) | Reader-facing time-boxed evaluation paths |
| [../../README.md](../../README.md) | Repository overview |
| [../../COLLABORATION.md](../../COLLABORATION.md) | Collaboration and permission model |
| [PUBLIC_ISSUE_INDEX.md](PUBLIC_ISSUE_INDEX.md) | Curated public issue map |
| [MAINTAINER_TRIAGE_PLAYBOOK.md](MAINTAINER_TRIAGE_PLAYBOOK.md) | Public issue triage rules |
