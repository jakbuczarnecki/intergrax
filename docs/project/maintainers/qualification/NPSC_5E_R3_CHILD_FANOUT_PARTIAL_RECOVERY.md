# NPSC-5E/R3 — Child & Fan-Out Partial Recovery

> **Status:** ACTIVE qualification (2026-09-10)

## Scope

Canonical partial recovery for child execution and fan-out topology: after partial failure, only the exact failed slot may recover; successful siblings remain preserved.

## Baseline

```text
R2 Final: c030d1d4752513bc11ec2597de4e760ae551a978
R3 qualified base: 1ab3771f8c49e13de0f85c47473522c7521f201a
```

## Production surface

| Module | Role |
|---|---|
| `intergrax/contracts/partial_recovery.py` | Typed recovery intent, disposition, policy |
| `intergrax/runtime/long_running/topology_recovery_snapshot.py` | Durable topology recovery snapshot |
| `intergrax/runtime/execution/fan_out_partial_recovery.py` | Fan-out partial recovery service |
| `intergrax/runtime/execution/orchestration_topology_submission.py` | `recover_failed_slot`, `restore_execution_record` |
| `intergrax/runtime/long_running/runtime_checkpoint.py` | Optional `topology_recovery` v2 field |

## Qualification

```text
tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py
```

## Checkpoint schema

```text
R2 CHECKPOINT CAPABILITY GAP: topology fan-out slot state
CAN BE ADDED COMPATIBLY TO v2: YES (optional topology_recovery field)
SCHEMA VERSION CHANGE REQUIRED: NO
```

## Key gates

- Successful sibling rerun: NO
- Failed slot only recovery: YES
- Nexus continuation port: YES
- R2 revision CAS: YES
- Cross-process: YES
- Second recovery runtime: NO
