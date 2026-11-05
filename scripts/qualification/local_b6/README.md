# EVOL-X — local B6 custody (Windows)

Production Ed25519 private keys **must not** live in the implementer repository.

## Required topology

| Role | Principal | Key / data |
|------|-----------|------------|
| Implementer | Developer / Cursor | Repo read-write; **no** production private key |
| Attestor | Separate Windows user (read-only git clone) | `EVOL_B6_ATTESTOR_READONLY=1`, `EVOL_B6_ATTESTOR_IDENTITY` |
| Custodian | Separate Windows user or offline HSM operator | `EVOL_B6_CUSTODIAN_MODE=1`, key under `C:\ProgramData\Intergrax\evol-b6-custody\` |
| Verifier | Any read-only | Public trust root + `evol_x_b6_evaluate_signed_attestation.py` |

Two agent sessions or two worktrees **do not** satisfy ECAP-P4 independence.

## Provisioning (administrator)

Run elevated once (adjust paths / usernames):

```powershell
# Option A — UAC launcher (recommended for unattended agent resume)
scripts/qualification/local_b6/Invoke-IntergraxEvolB6ElevatedLaunch.ps1 -RepoRoot <repo>

# Option B — direct admin shell
scripts/qualification/local_b6/Install-IntergraxEvolB6Custody.ps1
scripts/qualification/local_b6/Invoke-IntergraxEvolB6UnattendedBootstrap.ps1 ...
```

Windows SAM account names are limited to **20** characters; bootstrap uses `IntergraxEvolAttst` / `IntergraxEvolCustn` mapping to logical attestor/custodian roles.

Then:

1. Create Windows users `IntergraxEvolAttestor` and `IntergraxEvolCustodian` (or domain equivalents).
2. Deny implementer account write access to `C:\ProgramData\Intergrax\evol-b6-custody\`.
3. Generate Ed25519 key in custody folder (custodian user only).
4. Publish **public** trust root JSON outside repo (see template in `b6_technical_qualification/templates/`).
5. Maintain append-only replay ledger JSON on custodian path only.

## Attestor flow (read-only process)

```powershell
$env:EVOL_B6_ATTESTOR_READONLY = "1"
$env:EVOL_B6_ATTESTOR_IDENTITY = "named-independent-attestor-id"
uv run python scripts/qualification/evol_x_b6_local_attestor.py `
  --session-root .tmp/session/evol-x-local-b6-attestor `
  --verification-report .tmp/session/evol-x-r2-final-reconciliation/run-2026-11-04/independent_verification_report.json `
  --trust-root-ref tr-local-published `
  --key-id k-prod-1 `
  --output C:\ProgramData\Intergrax\evol-b6-custody\pending\unsigned.json
```

## Custodian flow (separate login)

```powershell
$env:EVOL_B6_CUSTODIAN_MODE = "1"
$env:EVOL_B6_CUSTODIAN_IDENTITY = "key-custodian-id"
uv run python scripts/qualification/evol_x_b6_custodian_sign.py `
  --unsigned C:\ProgramData\Intergrax\evol-b6-custody\pending\unsigned.json `
  --private-key C:\ProgramData\Intergrax\evol-b6-custody\private\k-prod-1.ed25519 `
  --replay-ledger C:\ProgramData\Intergrax\evol-b6-custody\replay_ledger.v1.json `
  --output C:\ProgramData\Intergrax\evol-b6-custody\signed\attestation.json
```

## Evaluate (does not publish B6 SHA)

```powershell
uv run python scripts/qualification/evol_x_b6_evaluate_signed_attestation.py `
  --attestation C:\ProgramData\Intergrax\evol-b6-custody\signed\attestation.json `
  --trust-root C:\ProgramData\Intergrax\evol-b6-custody\public\trust_root.v1.json `
  --replay-ledger C:\ProgramData\Intergrax\evol-b6-custody\replay_ledger.v1.json
```

Formal `B6_ACCEPTANCE_PUBLICATION_SHA` still requires governance `IN_FORCE` and authorized human publication commit.
