# LKW Model Runtime Portability Proof

- schema: `lkw.model_runtime_portability.proof.v2`
- proof_id: `lkw-model-runtime-proof:ea8f5bb82279`
- classification: controlled local-provider live LKW product proof for exact Ollama and vLLM provider/model pairs
- overall: **PASS**
- repository_head_at_proof: `7b889e700d7e0d7b4d008f20382d5497e075df3c`
- repository_head_role: `pre_evidence_commit_head`
- working_tree_classification: `task_owned_and_unrelated_changes`
- vllm_provisioning: `committed_compose_sufficient`

## Qualified provider pairs

### ollama
- configured_model: `qwen2.5:14b`
- resolved_model: `qwen2.5:14b`
- server_model: `qwen2.5:14b`
- adapter_class: `LangChainOllamaAdapter`
- server_version: `0.32.5`
- canonical_resolver: `True`
- http_ask_status: `200`
- ask_persisted: `True`
- failure_code: `None`
- server_model_digest: `sha256:7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`

### vllm
- configured_model: `Qwen/Qwen2.5-3B-Instruct`
- resolved_model: `Qwen/Qwen2.5-3B-Instruct`
- server_model: `Qwen/Qwen2.5-3B-Instruct`
- adapter_class: `VllmChatAdapter`
- server_version: `0.23.0`
- canonical_resolver: `True`
- http_ask_status: `200`
- ask_persisted: `True`
- failure_code: `None`

## Shared index

- workspace_id: `80cd9b6d-90b6-4a1f-9ffb-eb718c8aa3c9`
- collection_identity: `intergrax`
- source_id: `src:d2af80663ef954d5a7f831f4fd51fdf487a4e475fc3d1c9996387d82e34493f0`
- document_id: `lkwdoc:9a294c003dca9599f6b20a00c85a9ceb`
- content_hash: `sha256:1bce80deb06ac8161025daf32a9759210c410ad5fa5d30c0a9db417e678afc4b`
- chunk_count: `1`
- vector_count: `1`
- embedding_provider: `hf`
- embedding_model: `sentence-transformers/all-MiniLM-L6-v2`
- embedding_dimensions: `384`

## Index invariance

- embedding_identity: `PASS`
- collection_identity: `PASS`
- vector_count: `PASS`
- source_identity: `PASS`
- document_identity: `PASS`
- content_hash: `PASS`
- chunk_count: `PASS`
- no_reindex: `PASS`

## Limitations

- exact configured Ollama and vLLM pairs only; not universal model parity
- runtime hot swapping not required or proven
- qwen2.5:7b was not the qualified full-product Ollama model