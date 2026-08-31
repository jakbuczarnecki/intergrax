# TOOLS-SIDE-EFFECT-SAFETY — Brutal Docker E2E Proof

Platform-native proof for idempotent side-effect safety with an **external Postgres oracle**.

## One command

From repository root:

```bash
docker compose -f tests/system/tools_side_effect_safety/docker-compose.yml up --build --exit-code-from proof-runner
```

Teardown:

```bash
docker compose -f tests/system/tools_side_effect_safety/docker-compose.yml down -v
```

## Topology

```text
proof-runner
  ├─ runtime-single (SQLite volume, DURABLE_SINGLE_HOST)
  ├─ runtime-a / runtime-b (Redis, SHARED_MULTI_HOST via RedisIdempotencyStore)
  ├─ effect-proxy (Toxiproxy) → external-effect-service → effect-postgres
  └─ docker.sock (process-kill scenarios)
```

## Production call path

```text
POST /invoke (runtime worker)
  → RuntimeToolInvoker.invoke
  → declarative governance / HITL (when configured)
  → IdempotencyPreEffectCoordinator.before_external_effect (claim)
  → RegistryToolExecutor → ChargeHandler (httpx)
  → external-effect-service POST /charge
  → Postgres effects + effect_attempts (oracle)
  → IdempotencyPreEffectCoordinator.after_external_effect / uncertainty handling
```

## Idempotency providers

| Topology | Provider |
|---|---|
| PROCESS_LOCAL | InMemoryIdempotencyStore (not used in this proof) |
| DURABLE_SINGLE_HOST | SQLiteIdempotencyStore on Docker volume (`runtime-single`) |
| SHARED_MULTI_HOST | RedisIdempotencyStore via `create_redis_idempotency_store` (`runtime-a`/`runtime-b`) |

`resolve_reference_idempotency_store(SHARED_MULTI_HOST)` returns `None`; multi-host runtimes inject the canonical Redis provider at composition time (production pattern).

## Artifacts

Proof JSON report:

`.tmp/session/tools-side-effect-safety-proof/docker-run/proof-report.json`

## Focused regression (host, before/after Docker)

```bash
uv run pytest \
  tests/unit/runtime/tools/test_tools_side_effect_safety.py \
  tests/unit/agents/persistence/test_declarative_claim_coordination_r3.py \
  tests/unit/agents/persistence/test_declarative_replay_semantics_r4.py \
  tests/unit/agents/test_runtime_context_lifecycle_r6b.py \
  tests/unit/runtime/tools/test_pcm_side_effect_coordination.py -q
```
