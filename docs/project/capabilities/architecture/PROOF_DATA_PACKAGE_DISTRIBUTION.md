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

## Related documents

- [DATASET_REPRODUCIBILITY.md](../../../platform_proofs/scenarios/verified_product_identification/DATASET_REPRODUCIBILITY.md)
- [DATASET_DISTRIBUTION_REVIEW.md](../../../platform_proofs/scenarios/verified_product_identification/DATASET_DISTRIBUTION_REVIEW.md)
