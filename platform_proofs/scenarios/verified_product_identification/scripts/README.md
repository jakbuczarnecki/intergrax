# VPI operational scripts

All batch CLIs, qualification runners, and operator launchers live under `scripts/<domain>/<purpose>/`. Do not add new `run_*.py` or ad-hoc tooling at the scenario package root (except the platform scaffold `run_proof.py`).

## Domains and purposes

| Domain | Purpose folder | Responsibility |
|--------|----------------|----------------|
| `dataset` | `dataset_generation` | WDC build, data-pack build, embedding materialization, portable package install |
| `dataset` | `dataset_validation` | Full data-pack validation CLI |
| `dataset` | `diagnostics` | Dataset / data-pack qualification and profiling |
| `dataset` | `operator_lifecycle` | Long-running data-pack resume and launch config |
| `embedding` | `diagnostics` | Embedding arena, qualification, bounded CUDA / representation checks |
| `storage` | `operator_lifecycle` | Storage bootstrap operator entry |
| `storage` | `storage_tooling` | Vector DB round-trip and storage qualification |
| `proof` | `proof_tooling` | Proof-scale dataset runners (e.g. proof-50) |
| `migration` | `migration` | Short-lived format upgrades (remove when done) |

Reserved purpose names (use when a script appears): `retrieval_tooling`, `identity_tooling`, `experimental`.

## Adding a new script

1. Pick **domain** (dataset, embedding, storage, proof, retrieval, identity, operator, migration).
2. Pick **purpose** from the table above.
3. Place `run_<action>.py` or `<domain>_<action>.py` in `scripts/<domain>/<purpose>/`.
4. Wire `python -m platform_proofs.scenarios.verified_product_identification.scripts.<domain>.<purpose>.<module>` from docs or `.bat` launchers.
5. Do **not** commit new modules under deprecated flat folders (`scripts/build`, `scripts/diagnostics`, `scripts/operator`) except thin re-export shims.

## Deprecated import paths

Stage 1 flat folders remain as **compatibility shims** only:

- `scripts/build/*` → `scripts/dataset/*` or `scripts/proof/*`
- `scripts/diagnostics/*` → `scripts/embedding/*` or `scripts/dataset/*` or `scripts/storage/*`
- `scripts/operator/*` → `scripts/dataset/operator_lifecycle/*` or `scripts/storage/operator_lifecycle/*`

`dataset/operator/*` at the dataset library root mirrors the resume modules for older import paths.

## Do not put at scenario root

- Operational `*.py` (builders, validators, bootstrap, qualification)
- Duplicate copies of scripts already under `scripts/`

Allowed at root: `run_proof.py` (platform proof entrypoint), `application/`, `proof/`, libraries (`dataset/data_pack/`, etc.), and documentation.
