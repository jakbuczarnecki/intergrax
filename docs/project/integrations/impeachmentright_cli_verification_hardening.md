# ImpeachmentRight - CLI Verification Hardening Audit

**Date:** 2026-07-21  
**Branch:** `development`  
**Issue:** [#190](https://github.com/jakbuczarnecki/intergrax/issues/190)  
**Scope:** CLI / packaging / verification hardening (FH-1…FH-15)  
**Base HEAD (pre-change):** `89b62203fd1a5f4d2d32914090d81b208dd739d2`

---

## Verdict

```text
ALL PUBLIC COMMANDS VERIFIED
```

---

## Public command matrix

| Command | Exit | Observed output (summary) | Key source | Network | Provider called | Publicly safe |
|---------|------|---------------------------|------------|---------|-----------------|---------------|
| `intergrax --help` (+ demo/receipt/external-work helps) | 0 | ASCII help; no `UnicodeEncodeError` under cp1252 | n/a | no | no | yes |
| `demo governed-contractor --offline --store build/external_work_demo` | 0 | `verification_valid: true`, relative `receipt_path` / `verification_key_path` / `verification_command` | n/a (signs in-process) | no | fake only | yes |
| `receipt verify …/accept_receipt.json --store build/external_work_demo` | 0 | `valid: true`, `key_source: store` | store | no | no | yes |
| `receipt verify` (no key source) | !=0 | `verification_key_source_required` | none | no | no | yes |
| `demo … --simulate-signing-failure --store build/external_work_recovery_demo` | 0 | `EXECUTION_SUCCEEDED_ATTESTATION_FAILED`, `receipt_path: null`, `recovery_command` | n/a | no | fake create+accept once | yes |
| `external-work retry-attestation exec-offline-accept --store …_recovery_demo` | 0 | `attested`, `provider_invoked: false`, `verification_valid: true` | demo_mode / deterministic demo signer | no | **no** | yes |
| `receipt verify …_recovery_demo/export/accept_receipt.json --store …` | 0 | `valid: true` | store | no | no | yes |
| second `retry-attestation` | 0 | `attested_idempotent`, `provider_invoked: false` | demo_mode | no | **no** | yes |

`provider_calls.json` after failure + two retries: `create_calls=1`, `accept_calls=1`, `cancel_calls=0`.

---

## Receipt portability

```text
demo process exited
fresh verifier process
store-backed public key resolver
valid receipt
mutated receipt rejected
```

Confirmed by `test_cli_verification_hardening.py` (subprocess `uv run intergrax`) and manual FH-15 commands. Mutated event yields `valid: false` + `digest_mismatch` + exit != 0.

---

## Recovery proof

```text
provider execution succeeded
signer failed
GER persisted
process restarted
retry attestation only
provider calls unchanged
receipt valid
second retry idempotent
```

RefuseProvider stub in retry CLI raises if any provider method is invoked. Ledger file is unchanged across retries.

---

## Security statement

```text
no private key in receipt
no private key in receipt store
no implicit key reconstruction from key_id
verifier never signs
```

- Implicit `key_id` → deterministic attestor reconstruction **removed**.
- Exactly one of `--store` / `--public-key-hex` / `--public-key-file` / `--demo-key`.
- Store keys: `keys/<key_id>.json` public material only (`FilesystemHostKeyResolver`).
- `--demo-key` is explicit and limited to `governed-contractor-offline-demo-1`.

---

## Tests executed

```text
43 passed - focused partner/platform/CLI suites
200 passed - execution_evidence + contracts + policy + adapter + host
```

Full `pytest -q` may still hit known Windows native crashes (Chroma/Kafka) unrelated to this stage.

---

## Final line

```text
READY TO PUBLISH PARTNER RESPONSE
```

Do **not** draft the `@impeachmentright` reply in this change set - publish packaging is complete; public wording remains an operator step.
