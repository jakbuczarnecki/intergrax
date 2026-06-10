# Implementation Journal Entry Template

Copy [`entries/_TEMPLATE.md`](entries/_TEMPLATE.md) to `entries/YYYY-MM-DD-<scope>-<slug>.md` and fill every section.

**Language:** English only. **Do not** duplicate plan tables or architecture canon — link instead.

---

## Frontmatter fields

| Field | Rule |
|-------|------|
| `id` | `IJ-YYYY-MM-DD-NNN` — see ID assignment below |
| `date` | Completion date (`YYYY-MM-DD`) |
| `tiers` | YAML list: one or more of `tier-0`, `tier-1`, `tier-2`, `tier-3` |
| `scope` | Domain basename (`RAG`, `TOOLS`) or path (`agents/<slug>`, `applications/<pkg>`, `intergrax/applications/_shared`) |
| `plan_ref` | YAML list of **formal IDs only** (e.g. `M-RAG.23`, `TOOL-ENG-4`, `AUDIT-IDEAL-14.3`, `K.1`) |
| `status` | `completed` |
| `commit` | Short git hash or `pending` |
| `adr` | ADR path or `none — <rationale>` |

Use a **single** `tier` key only for backward compatibility with older entries; prefer `tiers` list.

---

## ID assignment (`NNN`)

1. Open [`INDEX.md`](INDEX.md).
2. Find rows with the same `date`.
3. Set `NNN` = highest existing suffix for that date + 1 (three digits, zero-padded).
4. Example: after `IJ-2026-06-10-007` → next is `IJ-2026-06-10-008`.

---

## Filename convention

```text
entries/YYYY-MM-DD-<scope>-<slug>.md
```

Examples: `2026-06-10-rag-m-rag-23.md`, `2026-06-10-tools-tool-eng-4.md`, `2026-06-10-agents-vendor-discovery-k1.md`

---

## INDEX row (prepend — newest first)

Insert **below** the table header row in [`INDEX.md`](INDEX.md):

```markdown
| IJ-YYYY-MM-DD-NNN | YYYY-MM-DD | tier-0 RAG | M-RAG.23 | [query expansion wiring](entries/2026-06-10-rag-m-rag-23.md) | `94bea682` |
```

Do **not** append at the bottom — the index is reverse-chronological.
