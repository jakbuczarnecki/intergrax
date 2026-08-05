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
