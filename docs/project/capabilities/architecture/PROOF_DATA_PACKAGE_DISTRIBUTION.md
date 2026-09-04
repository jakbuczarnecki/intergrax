# Proof Data Package Distribution

**Status:** Canonical (VPI-IMPLEMENTATION-5C3)  
**Schema:** `intergrax.proof_data_package.v1`  
**Owner:** Intergrax platform proof infrastructure

## Purpose

Large scenario proofs require multi-gigabyte external assets (datasets, precomputed embeddings, manifests). These assets must be distributed independently from `proof.json` v3 discovery descriptors and independently from Git.

This module provides reusable, provider-neutral primitives:

| Component | Responsibility |
|-----------|----------------|
| `ProofDataPackageDescriptor` | Immutable package identity, file list, SHA256 + size, redistribution status |
| `DataPackageTransportPort` | Obtain bytes (HTTP reference, local file mirror for tests) |
| `DataPackageCache` | Content-addressable cache keyed by SHA256 |
| `DataPackageInstaller` | Validate descriptor, download/resume, verify, atomic publish |

**Non-goals in generic layer:** VPI semantics, PostgreSQL/Qdrant, embedding execution, cloud SDK uploads, proof.json v3 changes.

## Package identity vs location

Package identity (`package_id`, `package_version`, file checksums) is separate from download location. The same immutable package may be mirrored at multiple HTTPS bases (R2, S3, GCS, Azure Blob, B2) without changing checksums.

Install requests supply `base_uri` separately from the committed descriptor.

## Trust model

The committed descriptor in the Intergrax repository is the trust anchor for expected file SHA256 values. Installers verify every byte against the descriptor; internal dataset/embedding manifests retain semantic identity responsibilities.

Cryptographic package signing is **not** implemented in 5C3. SHA256 provides transport integrity only.

## Operations

| Operation | Command / entry |
|-----------|-----------------|
| **Build descriptor** | Scenario-owned builder (`data_package/build_descriptor.py`) from trusted local files |
| **Install package** | `setup_data.py` (VPI) or `DataPackageInstaller` API |
| **Publish** | Future infrastructure upload; blocked until redistribution review |

## VPI integration

VPI-specific semantics live under `platform_proofs/scenarios/verified_product_identification/data_package/`. Storage bootstrap resolves installed paths when `data_package/installed/` is populated; it does **not** trigger network download.

## HTTP transport resume semantics

`HttpDataPackageTransport` derives the resume offset from the current partial file size before **every** HTTP attempt. Caller-supplied `resume_from_byte` is a hint only; on-disk partial bytes are authoritative and are never truncated to match stale caller state.

| Behavior | Policy |
|----------|--------|
| Transient network failure | Partial file is preserved; next attempt resumes from latest persisted byte |
| `Content-Range` mismatch on 206 | Fail closed; never append mismatched bytes |
| Server ignores `Range` (200) | Reset only the affected partial file and restart from zero |
| HTTP 416 | Bounded single reset-and-restart for that file; no unbounded recursion |
| Retries | `max_retries` is the total HTTP attempt budget per `download_file` call |
| Diagnostics | Error messages sanitize URIs (query strings omitted) |

`TransportDownloadResult.bytes_written` reports useful bytes persisted during the invocation (excluding bytes discarded when a partial is reset). `DataPackageInstallReport.bytes_downloaded` aggregates those per-file persisted byte counts across the install.

## Related documents

- [DATASET_REPRODUCIBILITY.md](../../../platform_proofs/scenarios/verified_product_identification/DATASET_REPRODUCIBILITY.md)
- [DATASET_DISTRIBUTION_REVIEW.md](../../../platform_proofs/scenarios/verified_product_identification/DATASET_DISTRIBUTION_REVIEW.md)
