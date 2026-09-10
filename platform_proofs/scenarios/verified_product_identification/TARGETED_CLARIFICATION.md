# Targeted clarification requirement selection (5C11)

## Purpose

5C11 selects **typed clarification requirements** from a `ProductIdentificationDecision` (5C10). It answers: *what exact missing fact should we ask for next?*

5C11 does **not** generate natural-language questions, templates, localization, or LLM output. Question rendering belongs to a later UI layer.

## Core invariant

Ask only for information that can **change the terminal decision**: eliminate a competing identity, resolve a required constraint gap the user can fill, or establish a material identifier the user may know.

Targeted clarification is **not** missing-field enumeration.

## Inputs and outputs

- **Input:** `ClarificationSelectionRequest` with `ProductIdentificationDecision` and `ProductIdentificationQueryContext`.
- **Output:** `ClarificationSelectionResult` with either `clarification_required=False` and a typed `NoClarificationReason`, or `clarification_required=True` with one **primary** requirement and deterministic **alternates**.

## Outcome handling

| Outcome | Clarification |
|---------|----------------|
| `VERIFIED` | None (`DECISION_ALREADY_TERMINAL`) |
| `NO_MATCH` | None (`DECISION_ALREADY_TERMINAL`) |
| `AMBIGUOUS` | Discriminator from `ambiguity_candidates` and `SourceIdentityFact` rows |
| `INSUFFICIENT_INFORMATION` | Only when missing knowledge is user-resolvable; catalog-only gaps do not ask redundant questions |

## USER vs CATALOG missing requirements

- **USER** origin: strong clarification candidate when material and answerable.
- **CATALOG** origin: never asked blindly; if the user already supplied the attribute in `required_constraints`, return `CATALOG_EVIDENCE_ONLY_GAP`.

## Discriminator discovery

For unresolved competitors, build a bounded view of `SourceIdentityFact` values per hypothesis. Valid discriminators have at least two distinct known values across unresolved hypotheses, exclude user-known attributes, ignore weak retrieval dimensions, and exclude source-local identifiers (SKU / product_id) as global discriminators.

## Selection ordering (deterministic)

1. USER-origin missing distinguishing requirement (material, answerable)
2. Complete-coverage material attribute discriminator
3. Complete-coverage identifier discriminator (when answerable)
4. Partial-coverage material attribute
5. Partial-coverage identifier

Tie-break: eliminable hypothesis count, distinct value count, known coverage, canonical attribute key ASC, `requirement_id` ASC.

**User-answerable material attributes** are preferred over technical identifiers (e.g. GTIN) for the primary requirement unless the query is identifier-oriented.

No weighted scores, probabilities, or confidence.

## Policies

- **Answerability:** `DeterministicClarificationAnswerabilityPolicy` — conservative classes; source-internal identifiers are not selected.
- **Materiality:** scenario allow-list aligned with VPI fixtures (`capacity`, `interface`, …).

## Provenance

Each requirement retains affected hypothesis IDs, supporting `SourceIdentityFact` rows, and deterministic `candidate_values` when known.

## Layering

Pure in-memory application code under `application/clarification/`. No dataset, data pack, storage bootstrap, providers, embeddings, or LLM.
