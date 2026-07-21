# LKW Platform Certification Matrix

Authoritative, machine-validated consolidation of current LKW
cross-platform certification evidence.

- Matrix ID: `lkw-platform-matrix-526905202f61`
- Matrix status: `VALID`
- Generated at (UTC): `2026-07-21T10:18:33Z`
- Generated from commit: `4847e9578251382c6a02eb8f00137dbf16b03553`

## Current certification status

The current shared LKW proof architecture is receipt-backed and live-certified
on native Windows and in a Linux Docker runtime. Native Linux host and macOS
runtime certification remain pending.

Application Hosting certification is not the same as complete multi-phase
Core Platform Proof certification.

| Profile | Environment | Implementation | Application Hosting | OS Interaction | Full Multi-Phase Core | Native Host Certified | Evidence |
|---------|-------------|----------------|---------------------|----------------|-----------------------|------------------------|----------|
| Windows native | native_host / windows | implemented | live-certified | live-certified | not certified by this profile | yes | yes |
| Linux Docker runtime | container / linux | implemented | live-certified | live-certified | not certified by this profile | no | yes |
| Linux native host | native_host / linux | implemented | not live-certified | not live-certified | not certified by this profile | no | no |
| macOS native | native_host / macos | implemented | not live-certified | not live-certified | not certified by this profile | no | no |

## Certified profiles

### windows_native_runtime

- Certification profile: `windows_native_runtime`
- Certification result: `PASS`
- Certification date: `2026-07-21T10:18:33Z`
- Source commit: `6b71a841c894728766fd6f574c9cd53ad12ec5f9`
- Application-hosting proof ID: `local_workspace:platform_application_hosting:lkw-hosting-a636dbd5d6e3`
- Interaction proof ID: `local_workspace:platform_windows_interaction:lkw-windows-interaction-2b42a6222d61`
- Source artifact: `docs/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json`
- Source artifact SHA-256: `2445b2222ba329169e8a437e5c4499f2835f559c1407f200e6d415e720519801`

### linux_docker_runtime

- Certification profile: `linux_docker_runtime`
- Certification result: `PASS`
- Certification date: `2026-07-21T09:34:20Z`
- Source commit: `40a73fbb455def6d5106180d74a7e65388457465`
- Application-hosting proof ID: `local_workspace:platform_application_hosting:lkw-hosting-a5cd37adf7d9`
- Interaction proof ID: `local_workspace:platform_linux_interaction:lkw-linux-interaction-44cd93cabccd`
- Source artifact: `docs/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json`
- Source artifact SHA-256: `bd940d683d445a08db5deff4f57d3071ed19890e821b875281371fb60ad33678`

## Implemented but not live-certified profiles

### linux_native_runtime

- Status: `NOT_CERTIFIED`
- Limitations:
  - Linux entrypoints are implemented.
  - No separate native Linux host live certification artifact exists.
  - Linux Docker runtime evidence does not certify native Linux installation.

### macos_native_runtime

- Status: `NOT_CERTIFIED`
- Limitations:
  - macOS entrypoints are implemented.
  - No macOS live certification artifact exists.
  - No macOS ProofReceipt has been recorded for this matrix.

## Evidence sources

```text
docs/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json
docs/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json
docs/public-adoption/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json
```

## Scope limitations

- The current shared LKW proof architecture is receipt-backed and live-certified on native Windows and in a Linux Docker runtime. Native Linux host and macOS runtime certification remain pending.
- Application Hosting certification is not the same as complete multi-phase Core Platform Proof certification.
- Linux Docker runtime evidence does not certify native Linux installation.
- macOS remains implemented but not live-certified.

## Reproduction and validation

Generate or refresh the matrix:

```bash
uv run python applications/local_workspace_application/scripts/generate-lkw-platform-certification-matrix.py
```

Check committed artifacts for staleness:

```bash
uv run python applications/local_workspace_application/scripts/generate-lkw-platform-certification-matrix.py --check
```

This matrix does not execute live proofs and does not create ProofReceipts.
It only aggregates and validates existing certification evidence.
