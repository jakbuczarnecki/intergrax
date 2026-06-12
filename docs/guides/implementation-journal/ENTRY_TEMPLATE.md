# Implementation Journal Entry Template

Create `entries/YYYY-MM-DD/` if needed; copy [`entries/_TEMPLATE.md`](entries/_TEMPLATE.md) to `entries/YYYY-MM-DD/<scope>-<slug>.md` and fill every section.

**Language:** English only. **Do not** duplicate plan tables or architecture canon — link instead.

---

## Frontmatter fields

| Field | Rule |
|-------|------|
| `id` | `IJ-YYYY-MM-DD-NNN` — see ID assignment below |
| `date` | Completion date (`YYYY-MM-DD`) |
| `tiers` | YAML list: one or more of `tier-0`, `tier-1`, `tier-2`, `tier-3` |
| `scope` | Domain basename (`RAG`, `TOOLS`) or path (`agents/<slug>`, `applications/<pkg>`, `intergrax/applications/_shared`) |
| `plan_ref` | YAML list of formal IDs — row (`M-RAG.23`), `AUDIT-IDEAL-X.Y`, named phase (`FAUDIT-32`), or backlog slot (`K.1`) — see README §plan_ref grammar |
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
entries/YYYY-MM-DD/<scope>-<slug>.md
```

Examples: `entries/2026-06-10/rag-m-rag-23.md`, `entries/2026-06-10/tools-tool-eng-4.md`, `entries/2026-06-10/agents-vendor-discovery-k1.md`

**Anti-patterns** (rejected by `scripts/check_implementation_journal.py`):

- `entries/2026-06-11-acp-close-pat-2.md` — date in filename; use folder `entries/2026-06-11/`
- `entries/acp-close-pat-2.md` — missing date folder
- Only `entries/_TEMPLATE.md` may sit directly under `entries/`

---

## INDEX row (prepend — newest first)

Insert **below** the table header row in [`INDEX.md`](INDEX.md):

```markdown
| IJ-YYYY-MM-DD-NNN | YYYY-MM-DD | tier-0 RAG | M-RAG.23 | [query expansion wiring](entries/2026-06-10/rag-m-rag-23.md) | `94bea682` |
```

Do **not** append at the bottom — the index is reverse-chronological.
