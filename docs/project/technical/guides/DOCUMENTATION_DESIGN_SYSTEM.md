# Documentation Design System

**Purpose:** Canonical standard for designing and modernizing Intergrax **domain architecture hubs** and **multi-layer feature architecture hubs**.

**Scope:** Structure, visual grammar, maturity/claim rules, navigation, and maintenance — not domain-specific content.

**Applies to:**

- `docs/project/architecture/<DOMAIN>.md`
- `docs/project/capabilities/architecture/<FEATURE>.md`

**Does not apply as a rewrite target in this phase:** satellites, plans, ADRs, root README, application product tours (except as reference examples).

**Related:** [DOCUMENTATION_MAP.md](../DOCUMENTATION_MAP.md) · [MATURITY_TAXONOMY.md](MATURITY_TAXONOMY.md) · [guides/README.md](README.md)

---

## 1. Canonical entry points

Intergrax does **not** create parallel public copies such as `PUBLIC_MEMORY.md` or `docs/project/platform/MEMORY.md`.

| Entry point | Path | Role |
|-------------|------|------|
| **Domain architecture hub** | `docs/project/architecture/<DOMAIN>.md` | Domain canon + public front section + router |
| **Multi-layer feature hub** | `docs/project/capabilities/architecture/<FEATURE>.md` | Cross-layer feature canon + public front section + router |

Each hub is simultaneously:

1. a readable **human-facing front section**,
2. the **main architecture canon** for its topic,
3. a **router** to satellites, plans, ADRs, guides, and proofs.

Satellites remain **extended technical depth** — not a public first-contact route.

---

## 2. Document taxonomy

One topic → one authoritative role. Do not duplicate whole sections across roles.

| Artifact | Owns | Does not own |
|----------|------|--------------|
| **Domain / feature hub** | Stable architecture, contracts, boundaries, human-facing explanation | Implementation tracker, decision rationale archive |
| **Satellite** | Bulky extended detail, deep module maps, long registers | Public first impression, maturity status alone |
| **Plan** | What is done / next, phases, gates, backlog rows | Architecture spec competing with hub |
| **Guide** | Authoring, operations, extension how-to | Domain canon |
| **ADR** | A specific decision and its trade-offs | Living system description |
| **Proof** | Bounded executable evidence | Product marketing claims |

**Hard rule:** Human-facing explanation describes stable architecture. Engineering canon holds exact technical semantics. Copy **links**, not paragraphs, across roles.

---

## 3. Layered mental model

Every hub follows this depth progression:

```text
HUMAN-FACING FRONT
        ↓
ARCHITECTURE EXPLANATION
        ↓
ENGINEERING CANON
        ↓
EXTENDED SATELLITES
```

A single hub file may contain the **first three** levels. Satellites are the **fourth** level and stay in `satellites/` (domain or feature tree per existing layout).

Do **not** remove from engineering canon:

- invariants and contracts,
- ownership rules,
- implementation boundaries,
- configuration semantics,
- qualification details.

Do **not** mix those with the reader's first screen.

---

## 4. Recommended canonical anatomy

Not every domain needs identical section titles. The **front section** must be consistent; later sections may follow domain-specific canon structure.

### 4.1 Human-facing front section (required pattern)

Use this order unless a subsection is genuinely not applicable (state why briefly, do not omit silently).

#### 1. Title + one-line definition

One clear sentence: **what this part of Intergrax is**.

- No internal implementation jargon in the first sentence.
- Optional subtitle after em dash for scope hint.

#### 2. Why it matters

Short prose: problem solved, why the platform needs it, value to developer / operator / organization.

#### 3. Claim / maturity boundary

If the domain is not fully production-qualified, state it **early** — callout or short note near the top.

- Use vocabulary from [MATURITY_TAXONOMY.md](MATURITY_TAXONOMY.md).
- Do not bury limitations at the end of the document.

#### 4. Primary audience

Include only when it changes how the document should be read (e.g. architects vs operators vs extension authors).

#### 5. At a glance

Compact table — pick rows that fit the domain, typically from:

| Typical row | Meaning |
|-------------|---------|
| Responsibility | What this domain owns |
| Key mechanisms | Named platform mechanisms (not file paths) |
| Boundaries | What is explicitly out of scope |
| Maturity | Current qualification / axis summary |
| Related systems | Neighbor domains or features |

Keep tables narrow. Prefer 4–7 rows, not exhaustive matrices.

#### 6. Flagship architecture visual

One primary diagram showing **where this subsystem sits in Intergrax and what role it plays**.

- Prefer hero SVG (see §6) for platform-position / marketing-relevant mental models.
- Mermaid is acceptable when the flagship idea is inherently flow-based.

#### 7. How it works

Plain-language walkthrough of the main process.

- Add a **single** Mermaid flow when the domain has a natural execution or lifecycle path.
- Numbered substeps optional for multi-stage flows.

#### 8. Responsibility / ownership boundaries

Explicit split:

- what this domain **owns**,
- what it **does not** own,
- what belongs to **applications (Tier-3)**,
- what belongs to **other Intergrax domains**.

Use subheadings or a compact table — not prose buried in a long paragraph.

#### 9. Relationship to Intergrax

Neighbor links in ecosystem terms (e.g. Memory ↔ Context Engineering ↔ RAG).

- Link to authoritative hubs; do not restate their canon.

#### 10. Extensibility

When the subsystem exposes plugin / adapter / provider / host extension surfaces — summarize surfaces and point to extension guides.

Omit section if no extension surface exists.

#### 11. Current maturity

One unambiguous maturity statement aligned with [MATURITY_TAXONOMY.md](MATURITY_TAXONOMY.md) and the domain plan.

- Hub states **current architecture truth + qualification boundary**.
- Plan holds phase rows and delivery status.

#### 12. Evidence / proof

Links to real proofs, qualification records, or bounded verification artifacts.

- No fabricated or aspirational proof claims.
- Separate **what the doc explains** from **what has been demonstrated**.

#### 13. Go deeper

Router table or list:

| Depth | Route |
|-------|-------|
| Engineering canon | Anchor below front section in same file |
| Satellites | `satellites/<DOMAIN>_*.md` |
| Plan | `docs/project/maintainers/plans/<DOMAIN>.md` or feature plan pair |
| ADR | Relevant `docs/project/technical/adr/entries/` |
| Guides | Operator / extension / audit slice guides |
| Proofs | `docs/project/proofs/` or qualification artifacts |

End the front section with a horizontal rule (`---`) before engineering depth begins.

### 4.2 Architecture explanation (optional bridge)

Short bridge between front section and deep canon when the hub needs a mid-level conceptual section (e.g. governance plane, three-layer model).

- One visual anchor.
- No duplicate of front-section boundaries.

### 4.3 Engineering canon (retain and reorganize, do not delete)

Existing deep sections remain authoritative:

- contracts and invariants,
- configuration semantics,
- module boundaries,
- qualification tables,
- operator-relevant defaults.

**Legacy hub problems to fix during modernization (DOC-3+):**

- Maintainer metadata, audit layers, and Cursor read-scope blocks **before** the reader understands the domain.
- Navigation inventory tables replacing a human intro.
- Walls of text with no visual anchor for the first several screens.

### 4.4 Maintainer / Cursor / agent material

Blocks such as:

- Cursor read scope and token budgets,
- audit instructions and layer IDs,
- hub metadata (plan link, last updated, phase tags),

**must not dominate the first screen.**

Place them:

- after the human-facing front section and a clear `---` divider, **or**
- in a labeled block such as `## Maintainer and Cursor context` / `## Cursor read scope (token budget)`.

Do not remove this material during DOC-3 — **relocate** it.

---

## 5. Domain hub vs feature hub

| Aspect | Domain hub | Feature hub |
|--------|------------|-------------|
| Path | `architecture/<DOMAIN>.md` | `capabilities/architecture/<FEATURE>.md` |
| Scope | Single platform domain pair | Cross-layer capability coordinating domain pairs |
| Front section | Domain responsibility and boundaries | Feature outcome and cross-domain coordination |
| Plan pair | `maintainers/plans/<DOMAIN>.md` | `capabilities/plan/<FEATURE>.md` |
| Canon rule | Domain pair remains authoritative for owning domains | Feature hub **coordinates** — does not replace domain canon |

Feature hubs use the **same front-section pattern**; "Relationship to Intergrax" emphasizes participating domains.

---

## 6. Visual design system

Documentation is a public engineering showcase. Choose the **simplest visual form that carries the idea**.

### 6.1 Flagship SVG / hero architecture diagram

**Use for:** primary mental model, platform position, representative workflow, presentation-quality architecture.

**Preferred markup when light/dark matters:**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="...-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="...-light.svg">
  <img alt="..." src="...-light.svg">
</picture>
```

**SVG requirements:**

- readable on GitHub at desktop and moderate mobile width,
- no microscopic text,
- accurate `alt` text,
- do not depict finished product UI unless it is real,
- co-locate under domain/feature `assets/` or established doc asset paths.

**Not required:** every domain needs its own SVG. Add only when it materially improves understanding.

### 6.2 Mermaid

**Use for:** execution flows, sequences, lifecycles, dependencies, ownership relationships, dynamic architecture.

**Rules:**

- one diagram → one idea,
- avoid giant graphs requiring zoom,
- prefer `flowchart LR/TB` or short `sequenceDiagram`,
- keep node labels short.

### 6.3 Tables

**Use for:** responsibilities, boundary matrices, maturity summaries, contracts, comparisons, capability matrices.

Avoid very wide tables (many columns). Split or move detail to satellites.

### 6.4 ASCII / text diagrams

**Not** the default public visual language.

Allowed only when:

- exceptionally simple,
- must be copy-paste friendly,
- showing a short logical path.

Otherwise prefer Mermaid or SVG. Legacy ASCII in engineering canon may remain until modernized; new front-section visuals should not default to ASCII walls.

---

## 7. Visual rhythm

Avoid:

- large text walls,
- many consecutive sections without a visual anchor,
- stacked tables with no prose between,
- three diagrams showing nearly the same thing,
- decorative graphics with no informational value.

Target rhythm:

```text
explanation
    ↓
visual
    ↓
short explanation
    ↓
table / flow
    ↓
technical depth
```

The document should read like **top-tier engineering platform documentation** — not a marketing landing page, not a raw internal runbook.

---

## 8. Content style

**Front section:**

- short sentences first; specialist terms after definition,
- no hype adjectives ("revolutionary", "groundbreaking"),
- show value, separate value from proof,
- link evidence instead of asserting it.

**Prefer:**

> Intergrax Memory keeps durable and session-scoped state available across execution boundaries.

**Avoid:**

> Intergrax introduces a groundbreaking revolutionary cognition substrate...

Convince through **precision and honesty**, not adjectives.

**Engineering canon:** precise MUST/MUST NOT language, stable identifiers, code paths where they clarify contracts — same bar as today, better placement.

---

## 9. Generated visual assets (DOC-3+ policy)

During hub modernization, new SVG/graphics are allowed **only when justified**.

| Rule | Detail |
|------|--------|
| Purpose | Every new diagram must have a stated informational goal |
| Reuse | Prefer existing assets when they already match the need |
| No duplication | Do not create a second diagram for the same idea |
| Light/dark pairs | Only when GitHub reading in both themes materially benefits |
| Accuracy | Visual must match architecture — no simplified fiction |
| Claim discipline | Graphics must not inflate maturity or proof status |

DOC-2 defines policy only — **no new assets in DOC-2**.

---

## 10. Definition of Done — hub quality gates

A hub is not complete when information is merely correct. It must pass all gates:

| Gate | Question |
|------|----------|
| **Readability** | Can someone new to the domain understand the first 2–3 screens? |
| **Architecture** | Is it clear where this subsystem sits in Intergrax? |
| **Visual** | Do the most important ideas have an appropriate visualization? |
| **Navigation** | Is it clear what is related and where to go deeper? |
| **Claim** | Are maturity and proof claims honest and bounded? |
| **Engineering** | Did the public front section preserve full canon precision below? |

---

## 11. Maintenance rules

| Change type | Update |
|-------------|--------|
| Implementation detail | Technical canon ± plan if delivery status shifts |
| Domain behavior or responsibility boundary | Human-facing front section **and** technical canon |
| Platform topology (new domain, moved ownership) | Affected hubs + [intergrax_runtime_architecture.md](../../architecture/intergrax_runtime_architecture.md) / platform overview as applicable |

Do not mirror full plan rows or ADR text into the hub. Update links and boundary statements instead.

---

## 12. Reference examples (not templates)

These documents illustrate **parts** of the standard. None is a complete target state alone.

| Document | Illustrates |
|----------|-------------|
| [GOVERNED_EXECUTION.md](../../architecture/GOVERNED_EXECUTION.md) | Human-facing opening, early maturity boundary, At a glance, responsibility split, flagship mental-model visual |
| [LKW_PRODUCT_TOUR.md](../../../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md) | Visual hierarchy, light/dark `<picture>` SVG, Mermaid flow, route / next-action pattern, explicit boundaries |

**Anti-patterns visible in legacy hubs (fix in DOC-3):**

| Pattern | Seen in | Fix |
|---------|---------|-----|
| Metadata block before definition | MEMORY, OBSERVABILITY | Move to maintainer section; write front section first |
| Navigation inventory as opening | RAG | Fold into "Go deeper"; lead with definition + boundary |
| Cursor read scope on screen one | MEMORY, OBSERVABILITY | Relocate after front section |
| Deep canon without front section | Most legacy domain hubs | Add §4.1 front section; retain canon below |

---

## 13. Modernization handoff (DOC-3)

To modernize a hub, an operator or agent prompt should reference this file explicitly, e.g.:

> Modernize `MEMORY.md` according to Intergrax Documentation Design System.

The session should know:

- how to structure the public entry,
- where technical depth starts,
- which visual types to use,
- how to treat satellites, plans, and proofs,
- how to state maturity without claim inflation,
- that Cursor/maintainer blocks are relocated, not deleted.

**Out of scope for DOC-3 unless separately requested:** changing code, satellites content rewrite unless cited, plan phase edits unless boundary changed.

---

## 14. Unresolved architectural decisions

None introduced by DOC-2. Existing DOC-1 topology (domain pairs, feature pairs, satellites, guides, plans, ADRs, proofs) is preserved.

If a future hub modernization reveals a conflict between this design system and [public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md](../../maintainers/public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md), escalate to maintainers — do not create a third documentation root.
