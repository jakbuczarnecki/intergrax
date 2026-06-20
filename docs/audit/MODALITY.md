# Modality (Vision, Audio, ML) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/MODALITY.md`](../architecture/MODALITY.md) · [`plan/MODALITY.md`](../plan/MODALITY.md)  
**Audit map layers:** 29 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with **full repository access**.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: MODALITY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Modality (Vision, Audio, ML) (`MODALITY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Modality (Vision, Audio, ML)** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **three modality planes** (A: LLM multimodal, B: ingest, C: deterministic ML): ToolRuntime surfaces, ModalityProfile, Celery/worker execution, cost/observability — no agent SDK bypass.

## Key symbols and contracts

ModalityProfile · VisionInferenceAdapter · ModelInferenceAdapter · VisionModelProfile · ModalityExecutionMode (CELERY) · AttachmentRef · tool_ids vision.*, speech.*, ml.*

## Active plan phases (verify status vs code reality)

W-ML harness Done · W-ML remote Triton/HF incremental · Phase W-ML registry extensions

## Known open gaps — re-validate every item (closed / still open / partial)

model_inference/ partial · remote serving incremental · Plane A vs C boundary discipline · online training out of scope

---

## 0. Context budget (mandatory — quality without bulk loading)

Deep audit = **targeted reads + code/gate evidence**, not loading entire plan files.

### Session rules
- **One domain per chat** unless the operator explicitly batches.
- **Never** read a file >500 lines in full — grep section headers, then `Read` with offset/limit.
- **Never** re-read the same file in one session unless it changed.
- Prefer **grep with path filters** over repo-wide semantic search for known symbols.
- Run **only** scripts in section 10 — no full-suite pytest unless this prompt lists a domain slice.
- Do **not** load `docs/audit_results/` unless RESUME/bootstrap says so.
- Respect **`.cursorignore`** — excluded paths are out of scope unless the operator points to them.

### Scoped plan read (`docs/plan/{DOMAIN}.md`)
Read **only**: `## 6.` open queue rows only · gap/remediation registers tied to **Known open gaps** and **Active plan phases** · skip `(closed)`, `(complete)`, `Archived` unless re-validating a listed gap

### Scoped architecture read (`docs/architecture/{DOMAIN}.md`)
Table of contents + sections for audit-map layers **29** + registers tied to **Known open gaps**. Skip historical paydown logs unless a gap ID points there.

### Scoped guide reads
- **Prefer** [`docs/guides/audit_slices/{DOMAIN}.md`](../guides/audit_slices/{DOMAIN}.md) — compact slice for this domain (replaces bulk IDEAL + AUDIT_MAP load)
- Otherwise: `IDEAL_HARNESS_AI_ARCHITECTURE.md` — sections for layers **29** only
- `INTEGRAX_HARNESS_AUDIT_MAP.md` — layers **29** + maturity §5 only
- `SYSTEM_INVARIANTS.md` — skim invariant IDs referenced in section 3 dimensions only

---


## 1. Canonical reads (scoped — in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — **layers 29 only** (see §0)
2. `docs/architecture/MODALITY.md` — **scoped sections** (see §0)
3. `docs/plan/MODALITY.md` — **scoped sections only** (see §0) — do **not** load the full file
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — **layers 29** + §5 maturity
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/llm_adapters/ (Plane A attachments)
intergrax/rag/document_loaders/ (Plane B ingest)
intergrax/multimedia/ · intergrax/model_inference/
intergrax/tools/providers/vision|speech|ml/
integrations/providers/speech_provider/
modality_celery_wiring.py · ThreadPoolModalityInferenceExecutor
intergrax/runtime/observability/modality_counters.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Plane C operations via ToolRuntime tools — not agent importing torch/onnx directly.
2. Plane A LLM vision attachments typed via AttachmentRef.
3. require_deterministic_cv forces Plane C not LLM vision guess.
4. Plane B ingest separate from Plane C inference (document_loaders vs model_inference).
5. Speech via IntegrationSpeechAdapter slugs — not vendor SDK in agent.
6. ModalityProfile caps: max_media_bytes, allowed_planes, vision_model_ids.
7. Celery broker path (INTERGRAX_MODALITY_CELERY_BROKER_URL) with thread-pool fallback.
8. Modality metrics on tool_invocation_end / TASK_COMPLETED.
9. V-COST fields populated for modality tool calls.
10. HF Hub not on hot path for production profile.
11. tool_ids Done status matches actual handler implementation.
12. Context budget policy caps media contribution to prompt.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Large image batch via worker pool vs Celery.
- Long audio transcription path.
- YOLO/ONNX in-process vs remote Triton.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ModalityProfile · ContextBudgetPolicy caps · integration speech_provider slugs · tts_voice_id

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/audit/README.md` §Shared production Harness checklist:

- Architecture & modularity
- Configuration & strategy selection
- Override & customization surfaces
- Observability, tracing & logging
- Security & governance
- Reliability & error handling
- Performance & scale
- Testing & verification
- Documentation alignment

---

## 7. Production baseline comparison

Compare against: **Triton/TorchServe/YOLO CV pipelines · Deepgram/ElevenLabs speech · ONNX edge · HF Inference Endpoints**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Agent imports cv2/torch directly · LLM vision for regulated CV when deterministic required · binary blobs inline in agent without AttachmentRef

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/model_inference/ -q
uv run pytest tests/unit/ -q -k modality
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/MODALITY.md` gap rows + `docs/architecture/MODALITY.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
