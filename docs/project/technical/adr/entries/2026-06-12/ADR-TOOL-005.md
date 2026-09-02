# ADR-TOOL-005: Hierarchical tool selection v1 and strategy plugin surfaces (TOOL-ENG-14,26,31)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-12 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/TOOLS.md`](../../architecture/TOOLS.md) · [`plan/TOOLS.md`](../../plan/TOOLS.md) TOOL-ENG-14,15,26,31,32 · ADR-TOOL-004 |

## Context

Large catalogs benefit from category-aware narrowing before LLM schema export. Canon describes hierarchical mode as multi-pass traversal (category → sub-schema → tool). TOOL-ENG-26/31 require custom `ToolSelectionStrategy` surfaces beyond enum routing.

## Decision

1. Add `ToolSelectionMode.HIERARCHICAL` and `HierarchicalToolSelectionStrategy`.
2. **v1 algorithm (deterministic):** `rank_categories()` by query overlap on category + tool metadata → bounded category pass budget (`tool_selection_max_hierarchy_passes`) → keyword rank tools within selected branches → top-k allow-list.
3. Defer **LLM category schema pass** (extra planner round-trip per branch) to a follow-up row - v1 closes category→tool narrowing without new prompts.
4. Add `RuntimeConfig.tool_selection_strategy` instance override (surface A) and `tool_selection_strategy_id` entry-point lookup via `intergrax.tool_selection_strategies` (surface B).
5. Emit `ToolSelectionDiagV1` (`ops:tool_selection`) on every ToolsStep selection resolve (TOOL-ENG-32).

**Rejected:** Reusing `skill_pack` as hierarchical mode. Blocking TOOL-ENG-14 on full LLM taxonomy pass before any shipped narrowing.

## Consequences

### Positive

- Category-aware L6 narrowing without catalog-wide schema export.
- Plugin surfaces A/B aligned with selection plugin canon.
- Selection telemetry for observability gates.

### Negative

- v1 is keyword-based category rank - not LLM category pick; operators must read ADR before expecting extra LLM passes.
- Entry-point group empty until host packages register strategies.

## Compliance

- Tier-0 metadata only (`ToolContract.category`).
- Tests: `test_hierarchical_tool_selector.py`, `test_tool_selection_registry.py`, `test_tool_selection_telemetry.py`.

## Implementation notes

- `intergrax/runtime/nexus/tools/hierarchical_tool_selector.py`
- `intergrax/runtime/nexus/tools/tool_selection_registry.py`
- `intergrax/runtime/nexus/tracing/tools/tool_selection.py`
