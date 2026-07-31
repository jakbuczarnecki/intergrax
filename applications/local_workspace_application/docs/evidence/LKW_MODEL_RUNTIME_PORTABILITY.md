# LKW Model Runtime Portability Proof

- schema: `lkw.model_runtime_portability.proof.v2`
- proof_id: `lkw-model-runtime-proof:aceb47c173c5`
- classification: controlled local-provider live LKW product proof for exact Ollama and vLLM provider/model pairs
- overall: **PASS**
- repository_head_at_proof: `053aedf6d046d75844ecc53a6a23e17c864c0508`
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

- workspace_id: `dc5f49cc-4c81-4fe8-8a7e-b6757ad3adb8`
- collection_identity: `intergrax`
- source_id: `src:4c585b88f67c40470e66814b774ed33794cd9c070496b76e643f54f16d3411fb`
- document_id: `lkwdoc:af59f50c59414f3c29de962cbf953179`
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