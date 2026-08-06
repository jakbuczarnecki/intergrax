# Vendor Knowledge publication permit

`DocumentStoreKnowledgeSyncPublicationFenceRepository` stores the lifecycle
fence and the optional `KnowledgeSyncPublicationPermitV1` in one authoritative
binding row. Permit acquisition replaces that row with one conditional
mutation; lifecycle enable, disable, detach and revision rotation use the same
CAS boundary. An unexpired permit makes lifecycle mutation fail with
`publication_in_progress`. An expired permit is cleared by the next successful
CAS mutation and can be recovered only for the still-current revision and
token.

The publication unit is one bounded page: sink delivery, remote-item state
receipt/index updates, checkpoint advancement, reconciliation CAS transitions,
and the terminal result are executed while the same permit is retained. The
source lease remains a separate prerequisite and controls sync concurrency;
the permit only linearizes lifecycle authority.

The permit is passed in `KnowledgeSyncBatch` and into fenced checkpoint,
remote-state and reconciliation repositories. Those repositories verify the
exact permit as defense in depth. A generic sink is not made transactionally
fenced by this field alone: Indexed Source sinks must verify the permit at
their visible commit boundary, while legacy sinks remain explicitly unfenced.
`require_fenced_publication=True` fails closed when the permit authority is
unavailable.

## Connected-source materialization commit

Connected-source indexing now separates durable prepared physical state from
query-visible state. Each staged document carries exact tenant/workspace/source
ownership, binding, delivery, remote ID and generation metadata, while the
receipt records the deterministic payload fingerprint and progresses through
`PREPARING` → `PREPARED` → `COMMITTED` (or `ABORTED`). Temporary materializer
files are never recovery state.

The query-visible authority for a bounded page is the single current
`ConnectedSourceMaterializationManifestV1` record. It contains deterministically
ordered remote/document entries, the materialization sequence, binding
configuration version, fence revision and only a SHA-256 fingerprint of the
lifecycle token. The page is bounded to 1,000 entries and 1 MiB serialized
size; oversized or malformed manifests fail closed.

The linearization point is the conditional insert/replace CAS of the
same-record lifecycle fence. Before that CAS, the immutable manifest, its
delivery index and per-remote candidate records, and an immutable publication
commit node are durable. The fence stores only the current V2 head; each node
contains the deterministic commit ID and predecessor ID, so later pages cannot
erase earlier committed history. Exact delivery reads traverse that bounded
chain, while remote resolution uses deterministic candidate keys and committed
sequence ordering. Per-remote active pointers are derived lookup accelerators
rebuilt after the fence CAS; missing or partial rebuilds cannot invalidate
manifest-backed visibility.

Immediately before the fence CAS, the coordinator callback verifies the exact
permit, lifecycle fence and still-owned source lease. A permit lost at that
boundary leaves prepared documents hidden. If the process crashes after the
CAS, the commit node is already durable, so retry only finalizes the derived
receipt and pointer state; no correctness-critical history write is required.
A completed receipt with a missing manifest or a malformed/missing chain node
is treated as uncertain and fails closed. Same-delivery payload mismatches and
same-sequence manifest mismatches are conflicts; lower sequences are stale and
higher sequences supersede the current manifest.

Immutable commit nodes and manifests are retention-ready only when no
authoritative head or retained descendant references them, remote-item active
history has been safely compacted, receipt/replay obligations have expired, and
the purge operation owns the exact binding. Retention and purge are outside
this authority.

`DELIVERY_MANIFEST` is the authority for new fenced connected-source
references. Historical `DELIVERY_RECEIPT` records retain the explicit
compatibility resolver; connected-source records are never silently treated as
`LEGACY_IMMEDIATE`. Tombstone activation remains deferred because this sink
still rejects deletion envelopes; missing page items are not interpreted as
deletions. The next task is ownership-scoped purge by tenant, workspace and
indexed-source binding.
