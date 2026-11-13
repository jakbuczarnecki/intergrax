# Artifact Lifecycle — architecture (multi-layer feature)

**Status:** implemented at the B-D8 implementation SHA (`T-IGX-036-Q`), draft PR, **not merged** before MCP-SERVER-X P8 (OD-PDC-SEQ-01); no certification claimed  
**Pair:** [plan/ARTIFACT_LIFECYCLE.md](../plan/ARTIFACT_LIFECYCLE.md)  
**Decision:** [`ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md`](../../maintainers/architecture/ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md) (ADR-PDC-08, Accepted by OD-PDC-ADR-01) and OD-PDC-SCOPE-04 = A  
**Kind:** a feature that coordinates domain owners. It is not a new domain: ownership of every artifact class stays with the domain that owns it.

## Purpose

Give every artifact a defined class, owner, identity, retention and access path without creating a global artifact storage engine. Before this feature the execution `ArtifactStore` was in-memory only, retention fields had no consumer, nothing spilled large tool output, and several types were named `ArtifactRef`.

## Taxonomy: four classes, four owners

| Class | Owner | Persistence |
| --- | --- | --- |
| `business` | Collaborative Work (`intergrax/collaborative_work`) | Collaborative Work repositories and `ArtifactContentStore` |
| `execution_evidence` | evidence plane (`intergrax/runtime/nexus/artifacts`) | execution `ArtifactStore` (in-memory, SQLite) |
| `context` | UCL (`intergrax/runtime/context_lifecycle`) | UCL repositories |
| `tool_output` | tool runtime (spill-to-reference) | execution `ArtifactStore`, ephemeral retention |

`intergrax/runtime/nexus/artifacts/taxonomy.py` declares the table and a consistency gate (one distinct owner per class). The execution store refuses `business` and `context` artifacts, so it cannot grow into a global engine. No global artifact storage engine, registry or scheduler is introduced.

## Shared reference contract

`intergrax/contracts/artifact_ref.py` keeps the v1 pointer `ArtifactRef` for run results and adds `DurableArtifactRef` (`durable_artifact_ref.v1`):

- **Mandatory content digest** (`sha256:<hex>`), validated at construction. There is no default and no empty value.
- **Producer identity** (`ArtifactProvenance`), **execution lineage** (`ArtifactLineage`: run, task, attempt, execution, trace, parent artifacts addressed by identity and digest), **tenant and workspace scope** (`ArtifactScope`) and a **retention class**.
- `reconstruct_artifact_lineage` rebuilds the ancestor chain from references and fails on a missing, altered or cyclic parent.
- **Migration:** `ArtifactRef.to_durable_ref` takes the digest from supplied content, an explicit digest or a platform-format checksum, and raises `ArtifactDigestRequiredError` when none is known. `Artifact` (the stored execution artifact) derives its digest from its bytes and rejects a supplied digest that does not match. The legacy `sha256=""` of the run artifact bundle is gone for sandbox files and is a typed error when projecting a bundle.

## Versioning, durable store, retention

- The execution `ArtifactStore` is append-only: `put` adds a version, nothing is overwritten, `get` returns the latest, `get_version` and `list_versions` return history.
- `SQLiteArtifactStore` follows the persistence-plane convention (`db_path`, WAL, `DURABLE_SINGLE_HOST`). Versions, retention class, expiry, producer, lineage and legal holds survive a restart; a tampered row fails its digest check on load.
- Retention class, TTL and legal hold are enforced by `retention.py`. `AUDIT_EVIDENCE` is never expirable. The sweep is a handler registered on the existing `TaskExecutionRegistry` and run by canonical execution; it skips evidence and legal hold and the store refuses their deletion as well. Compaction operates on context fragments only and never touches the store.

## One governed access port

`governed_access.py` is the single port in front of every store:

1. tenant check (the request tenant must equal the caller tenant; a mismatch never reaches a store);
2. authorization through the existing governance owner (`AgentRuntimeGovernancePort`: capability grant, policy, decision);
3. the operation on a storage-only read source (`sources.py`: execution store, Collaborative Work content);
4. digest verification of what is returned;
5. one `GovernanceAuditEvent` per operation outcome through the existing audit recorder, carrying actor, tenant, execution identity and artifact id.

Stores contain no security logic. The port adds no policy engine, no registry and no authority. Capabilities: `artifact.read`, `artifact.download`, `artifact.legal_hold`, `artifact.delete`.

## Governed secure download (OD-PDC-SCOPE-04 = A)

`issue_download_url` is the governed read port plus a short-lived provider URL: authorization, tenant check and audit first, then one `presigned_url(..., method="GET")` with a TTL clamped to the port maximum. No presigned URL is issued outside this port in the artifact lifecycle; a gate scans product code, and the agent-callable `storage.presigned_url` tool fails closed instead of signing.

## Spill and context references

`ToolOutputSpiller` writes tool output above a threshold to the execution store (`tool_output`, ephemeral retention, producer and lineage set) and returns a self-verifying `ArtifactReferenceToken` plus a short preview. A context reference is a mandatory context fragment carrying that token, so fragment compaction never drops it, and `ArtifactReferenceResolver` resolves it through the governed port and re-checks the digest.

## Dedupe

`intergrax/contracts/task_artifacts.py` is the single definition of the task artifact contracts; `intergrax/applications/contracts/application_artifacts.py` is a COMPAT-X shim that re-exports them. The trace alias `ArtifactRef = TraceArtifactRef` is no longer bound; `TraceArtifactRef` is the only name (the replay plane's own type is `ReplayArtifactRef`), and the old import still resolves through a module-level `__getattr__` (COMPAT-X shim).

## Boundaries

`intergrax/` imports neither `agents/` nor `applications/`. Stores import no governance module (gated). Business and context artifacts keep their own owners and stores.
