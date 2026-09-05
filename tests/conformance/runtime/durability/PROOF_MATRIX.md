# P0C-8 Durability Conformance — Proof Matrix

| Proof | Mechanism | Providers | Test |
| --- | --- | --- | --- |
| Redelivery same identity | BackgroundExecutionIdentityPersistence | KV, DocumentStore | `test_redelivery_preserves_identity_after_restart` |
| Retry A2 recovery | AttemptLifecycleService | KV, DocumentStore | `test_retry_attempt_survives_restart` |
| Multiple redelivery A2 | AttemptLifecycleService | KV, DocumentStore | `test_multiple_redelivery_after_restart_returns_a2` |
| Checkpoint resume identity | checkpoint/recovery authority | Checkpoint (SQLite) | `test_checkpoint_resume_identity_survives_restart` |
| Checkpoint attempt conflict | resolve_root_task_identity | Checkpoint | `test_checkpoint_explicit_attempt_conflict_denies_after_restart` |
| Recovery precedence | ExecutionTreeSnapshot authority | in-memory canonical | `test_recovery_precedence_survives_restart` |
| Recovery corruption | apply_runtime_checkpoint_to_graph | in-memory canonical | `test_recovery_corruption_fails_closed_after_restart` |
| Cancellation blocks resume | ExecutionTerminalService | Checkpoint | `test_cancellation_blocks_stale_checkpoint_after_restart` |
| Terminal blocks resume | ExecutionTerminalService | Checkpoint | `test_terminal_outcome_blocks_resume_after_restart` |
| Terminal winner restart | ExecutionTerminalService | KV, DocumentStore | `test_terminal_winner_survives_restart` |
| Terminal idempotency | ExecutionTerminalService | KV, DocumentStore | `test_failed_terminal_idempotent_after_restart` |
| Terminal redelivery skip | admit_background_execution_reentry | KV, DocumentStore | `test_terminal_redelivery_skips_handler_after_restart` |
| A2 + terminal combined | identity + attempt + terminal | KV, DocumentStore | `test_a2_with_cancelled_terminal_blocks_without_a3_after_restart` |
| Integrated lifecycle | single terminal authority across background + checkpoint/resume | KV/DocumentStore + Checkpoint | `test_integrated_lifecycle_a1_a2_terminal_restart_denies_execution` |
| Cross-component terminal authority | shared canonical store | KV, DocumentStore | `test_background_terminal_blocks_checkpoint_resume_without_second_commit` |
| Reverse terminal visibility | background sees Nexus-side commit | KV, DocumentStore | `test_checkpoint_or_nexus_terminal_blocks_background_redelivery_without_second_commit` |
| Terminal conflict | cross-consumer immutable authority | KV, DocumentStore | `test_shared_terminal_authority_rejects_cross_consumer_conflict` |
| Custom terminal provider | pluginable store backs all consumers | custom | `test_custom_terminal_store_can_back_all_consumers` |
| Composition gate | worker bootstrap shares terminal | n/a | `test_worker_bootstrap_does_not_construct_checkpoint_terminal_when_kv_authority_exists` |
| Tenant isolation | all authorities | KV, DocumentStore | `test_tenant_isolation_for_identity_attempt_and_terminal` |
| Provider namespace | BackgroundExecutionIdentityPersistence | KV, DocumentStore | `test_provider_namespace_isolation_after_restart` |
| Corruption fail-closed | all durable stores | KV | `test_corrupt_*` |
| Store unavailable | ExecutionTerminalService | KV | `test_store_unavailable_fails_closed_on_redelivery_after_restart` |
| Codec round trip | terminal persistence | KV, DocumentStore | `test_terminal_codec_round_trip_survives_restart` |
| Architecture gate | import hygiene | n/a | `test_conformance_suite_does_not_import_provider_transport_implementations` |

## Identity authority matrix

| Fact | Authority |
| --- | --- |
| transport mapping | BackgroundExecutionIdentityPersistence |
| active Attempt | AttemptLifecycleService |
| recovery state | checkpoint/recovery authority |
| terminal outcome | ExecutionTerminalService (one per composition graph) |
| invocation fence | IdempotencyStore |

## Provider qualification matrix

| Capability | KV | DocumentStore | Checkpoint |
| --- | ---: | ---: | ---: |
| background identity | yes | yes | n/a |
| attempt lifecycle | yes | yes | n/a |
| terminal authority | yes | yes | yes (standalone profile only) |
| checkpoint/recovery | n/a | n/a | yes |

## Terminal authority

Terminal provider is selected **per composition graph**. KV / DocumentStore / Checkpoint are interchangeable providers, not simultaneous authorities for the same execution lifetime.

## Restart semantics

Restart = same durable backing primitive + new adapter + new service instance (`fresh_admission_composition`, `fresh_checkpoint_composition`).

## Disclaimer

P0C guarantees durable lifecycle convergence, **not** exactly-once external side effects.
