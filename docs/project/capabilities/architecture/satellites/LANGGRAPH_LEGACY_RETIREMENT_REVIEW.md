<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LCI-8A — LangGraph legacy retirement review

**Review status:** `READY_FOR_REVIEW`  
**Starting HEAD:** `2de02fcc26a81fcecdb52521f8781abcdfb82049`  
**Validated HEAD:** `8771ea1ec0f2e6fad76aa4f4f6e414650381b0c0`  
**Remote:** `origin/development` at the same commit  
**Ancestors:** `c8efb2d13df82481f2ae6a6738dfb309b288e136` and
`2de02fcc26a81fcecdb52521f8781abcdfb82049` are ancestors of HEAD.

## Packaging and reachability

- `[project].dependencies` contains no `langgraph`.
- `langgraph-legacy = ["langgraph>=0.0.40"]` is an explicit optional extra.
- The reused LCI-7B evidence records zero installed `langgraph*` distributions
  and PASS for native core, Nexus and Harness smoke.
- The reused LCI-7C evidence records PASS for the `langgraph-legacy` extra and
  its legacy import boundary.
- Therefore a default Intergrax installation cannot reach the LangGraph
  package without explicit legacy opt-in. This is not an architecture
  regression.

## Canonical occurrence inventory

The exact mechanical scan covered `import langgraph`, `from langgraph`,
`langgraph.`, `StateGraph`, `CompiledStateGraph`, `add_messages`, and
`langgraph-legacy`. `CompiledStateGraph` had no matches.

| Occurrence family | Canonical paths / result | Classification |
|---|---|---|
| Native LangGraph-compatible skill-pack schema/importer | `intergrax/skills/importers/langgraph_skill_pack.py`; `intergrax/applications/_shared/skill_import_wiring.py`; public re-export in `intergrax/skills/importers/__init__.py` | `CANONICAL_RUNTIME` |
| Supervisor graph bridge | `intergrax/supervisor/supervisor_to_state_graph.py:198,220-221` (`END`, `StateGraph`) | `OPTIONAL_LEGACY_RUNTIME` |
| Web-search graph adapter | `intergrax/websearch/integration/langgraph_nodes.py:11,30` (`add_messages`) | `OPTIONAL_LEGACY_RUNTIME` |
| Repository tests and synthetic blocked-import fixtures | `tests/unit/architecture/test_langchain_boundary.py`, `tests/unit/architecture/test_langgraph_not_required.py`, `tests/unit/knowledge/contracts/test_document.py`, packaging assertions | `TEST_ONLY` |
| Guards, compatibility gate, grandfather register, and root packaging declaration | `scripts/maintenance/`, `scripts/ci/`, `scripts/maintenance/langchain_boundary_grandfather.json`, `pyproject.toml` | `TOOLING_ONLY` |
| Guides, architecture/plan records, inventory and installation receipts | Matching `docs/project/**` occurrences | `DOCUMENTATION_ONLY` |

The skill-pack importer parses a LangGraph-compatible JSON shape into the
native `SkillManifest`; it does not import or require the `langgraph` package.
It is a compatibility data format, not a LangGraph runtime owner.

## Generated runtime contexts

The guard reports four known generated import findings. They are derivatives,
not independent architectural owners:

| Generated path | Canonical source | State |
|---|---|---|
| `applications/local_workspace_application/docker/runtime-context/intergrax/supervisor/supervisor_to_state_graph.py` | `intergrax/supervisor/supervisor_to_state_graph.py` | Current; SHA-256 matches |
| `applications/local_workspace_application/docker/runtime-context/intergrax/websearch/integration/langgraph_nodes.py` | `intergrax/websearch/integration/langgraph_nodes.py` | Current; SHA-256 matches |
| `applications/lab_application/docker/runtime-context/intergrax/supervisor/supervisor_to_state_graph.py` | `intergrax/supervisor/supervisor_to_state_graph.py` | `STALE_GENERATED_COPY`; SHA-256 differs |
| `applications/lab_application/docker/runtime-context/intergrax/websearch/integration/langgraph_nodes.py` | `intergrax/websearch/integration/langgraph_nodes.py` | `STALE_GENERATED_COPY`; SHA-256 differs |

The corresponding generated packaging files are
`applications/local_workspace_application/docker/runtime-context/pyproject.toml`,
`.../uv.lock`, `applications/lab_application/docker/runtime-context/pyproject.toml`,
and `.../uv.lock`; these are also `GENERATED_RUNTIME_CONTEXT`, not owners.
`scripts/build/build_application_image.py` delegates to
`materialize_application_build_context`, which copies the canonical
`intergrax/` tree and root packaging into the selected context. A subsequent
materialization will therefore reproduce the two canonical LangGraph runtime
files in each context. No generated copy was edited.

## Caller and ownership inventory

- **Skill-pack importer:** `skill_import_wiring.py` is a direct optional
  production consumer for Product/Lab profiles; `__init__.py` re-exports the
  importer. The maintenance probe and architecture test are tooling/test
  consumers.
- **Supervisor:** no direct production or application caller was found.
  `intergrax/supervisor/__init__.py` has `__all__ = []` and describes the
  package as experimental and outside Tier-1 Nexus. The boundary test only
  creates a synthetic import fixture. The extension guide documents the
  direct `build_langgraph_from_plan` compatibility entry point.
- **Websearch:** no direct production or application caller was found.
  `WebSearchNode`, `websearch_node`, and `websearch_node_async` are
  self-contained in the adapter module. The compatibility installation gate
  imports the module as a tooling check.
- **Generated:** the four runtime-context files are copied artifacts, never
  independent callers.
- **Default production:** no Nexus, Harness, runtime, or application source
  imports LangGraph. The bounded runtime/harness search returned no matches,
  and the LCI-7B smoke independently passed.

The supervisor uses explicit lazy construction and directs users to Nexus /
`HarnessApplication`. The websearch adapter depends on the LangGraph-specific
`add_messages` reducer when available and otherwise supplies an identity
fallback; its search behavior itself delegates to native `WebSearchExecutor`.

## Public compatibility and native replacement

| Surface | Canonical path | Consumers | Default reachable | Optional extra | Native replacement | Decision |
|---|---|---|---:|---|---|---|
| LangGraph-compatible skill-pack import | `intergrax/skills/importers/langgraph_skill_pack.py` | Product/Lab profile wiring; public importer export | NO package dependency | None | Native `SkillManifest` conversion already exists | `KEEP_OPTIONAL` |
| Supervisor `StateGraph` bridge | `intergrax/supervisor/supervisor_to_state_graph.py` | No in-repo production caller; documented external compatibility path | NO | `langgraph-legacy` | Nexus loop, `AgentGraph`, and `HarnessApplication` cover default orchestration, not this ABI | `KEEP_OPTIONAL` |
| Websearch LangGraph node wrapper | `intergrax/websearch/integration/langgraph_nodes.py` | No in-repo production caller; legacy direct-import compatibility | NO | `langgraph-legacy` | Native `WebSearchExecutor` covers search, but not the LangGraph state/reducer ABI | `KEEP_OPTIONAL` |

The supervisor is not package-exported, but its direct module path is
documented as an optional compatibility entry point. The websearch module
has no package-level public re-export, but its module and functional wrappers
explicitly preserve backward compatibility and are used by the 7C boundary
probe. These facts create compatibility risk even without repository-local
callers. The skill-pack importer is actively profile-gated and publicly
re-exported.

## Decision

**`KEEP_OPTIONAL`**

LangGraph is not required by the default installation or by Nexus, Harness,
agent execution, or the default application runtime. Retaining the three
surfaces has bounded cost because the package is isolated behind the explicit
`langgraph-legacy` extra where a runtime dependency exists, while the
skill-pack path has no package dependency. Immediate removal would break an
active Product/Lab compatibility format and documented or backward-compatible
legacy entry points without an in-repository migration need.

The bounded follow-up is maintenance, not removal: keep the extra and lazy
boundary, keep the inventory/guard grandfather records synchronized, and
regenerate runtime contexts through the canonical builder when those
applications are rebuilt. Any deprecation or removal requires a separate,
explicit product decision and task; LCI-8A does not start it.

## Evidence checks

- LangGraph guard: exit `1` only because of the four known generated findings;
  canonical findings `0`, new findings `0`.
- `validate_langchain_inventory.py`: PASS (`69` IDs, `0` unclassified).
- `check_langchain_boundary.py`: PASS; `0` new forbidden, `0` stale entries.
- `uv lock --check`: PASS.
- `git diff --check`: PASS.
- Production change budget: `0`; RAG paths untouched.