# Governed External Work — Host Lifecycle

**Status:** Platform completion (PC-5…PC-8) — 2026-07-21  
**Role:** Host orchestration, persistence, attestation recovery, offline demo  
**Upstream GEC boundary:** [`governed_external_execution.md`](governed_external_execution.md)  
**Attestation:** [`execution_evidence_and_host_attestation.md`](execution_evidence_and_host_attestation.md)

---

## Ownership

| Layer | Owns |
|-------|------|
| Tier-2 | Mapping, policy request composition, provider execution, `GovernedProofProfile` |
| Host | Lifecycle orchestration, `ExecutionBoundaryEvent`, signing, receipt, persistence, recovery |
| Runtime / Nexus | `task_id`, `run_id`, continuation interrupt identity |
| Verifier | Offline validation only — no authorization, no execution |

---

## Atomic result path

```text
RuntimePolicyBundleEvaluator(ImmutableRuntimePolicyBundle)
  → EvaluatedPolicyDecision
ProviderInvocation (created before provider call)
  → provider mutation
  → ProviderInvocationOutcome
  → GovernedProofProfile
  → GovernedExecutionResult   (atomic consistency)
  → governed_execution_boundary_event.v1
  → HostAttestation
  → execution_evidence.proof_receipt.v1 (+ policy_bundle_artifact)
```

---

## Host states

`REQUESTED` · `CREATE_POLICY_DENIED` · `CREATE_IN_PROGRESS` · `QUOTE_RECEIVED` ·
`AWAITING_HUMAN` · `AWAITING_PAYMENT` · `ACCEPT_POLICY_DENIED` ·
`EXECUTION_IN_PROGRESS` · `EXECUTION_FAILED` ·
`EXECUTION_SUCCEEDED_ATTESTATION_PENDING` ·
`EXECUTION_SUCCEEDED_ATTESTATION_FAILED` ·
`EXECUTION_SUCCEEDED_ATTESTED` · `CANCELLED`

Provider status → host state mapping is explicit and provider-neutral
(`map_provider_status_to_host_state`). Providers are not assumed to share one
state machine.

---

## Persistence ports

| Port | Purpose |
|------|---------|
| `GovernedExecutionStore` | `GovernedExecutionResult` + host state + optional EBE JSON |
| `ProofReceiptStore` | Portable receipts (no private keys) |
| `PolicyBundleArtifactStore` | Immutable pack bodies |
| `ContinuationStateStore` | Surfaced continuation requests |

Local implementations: in-memory (unit tests) and filesystem (offline demo).

---

## Attestation-only recovery

```text
execution succeeded → result persisted → signing failed
  → EXECUTION_SUCCEEDED_ATTESTATION_FAILED
  → retry_attestation(execution_id)
  → no CREATE/ACCEPT/CANCEL provider call
```

Idempotent when a receipt already exists.

---

## Official offline demo

### Standard offline demo

```bash
uv sync

uv run intergrax demo governed-contractor \
  --offline \
  --store build/external_work_demo

uv run intergrax receipt verify \
  build/external_work_demo/export/accept_receipt.json \
  --store build/external_work_demo
```

Demo exports `keys/<key_id>.json` (public verification material only) and prints
`verification_key_path` + `verification_command`. Receipt JSON does **not** embed
the public key. Verifier requires exactly one explicit key source:
`--store`, `--public-key-file`, `--public-key-hex`, or `--demo-key` (local/test only).

### Mutation rejection

```bash
uv run intergrax receipt verify \
  build/external_work_demo/export/accept_receipt_mutated.json \
  --store build/external_work_demo
```

Expect `valid: false`, `digest_mismatch`, exit != 0.

### Signer failure and recovery

```bash
uv run intergrax demo governed-contractor \
  --offline \
  --simulate-signing-failure \
  --store build/external_work_recovery_demo

uv run intergrax external-work retry-attestation \
  exec-offline-accept \
  --store build/external_work_recovery_demo

uv run intergrax receipt verify \
  build/external_work_recovery_demo/export/accept_receipt.json \
  --store build/external_work_recovery_demo
```

`retry-attestation` signs only from persisted GER/EBE — no provider side effect.
`provider_calls.json` in the demo store stays unchanged across retries.
Second retry returns `reason: attested_idempotent`, `provider_invoked: false`.

Also: `external-work demo-create` / `demo-accept` / `show` / `receipt`.

---

## Invariants

- continuation ≠ execution  
- human evidence ≠ authorization  
- payment evidence ≠ authorization  
- receipt ≠ authorization  
- verification ≠ authorization  
- Each CREATE / ACCEPT / CANCEL requires its own policy evaluation and `invocation_id`.

---

## Key management (PC-10)

- `HostAttestor` - signing boundary
- `HostKeyResolver` / `HostKeyMetadataProvider` - verification keys, algorithm allowlist, deprecated keys
- `FilesystemHostKeyResolver` - offline store-backed public keys under `keys/<key_id>.json`
- Local Ed25519 / `--demo-key` = test/local reference only; production signer = DI (`HostAttestor`); no remote KMS/HSM in-tree
- Receipt store and keyring never contain private keys, seeds, or mnemonics
- `key_id` alone is never enough to reconstruct a verification or signing key
