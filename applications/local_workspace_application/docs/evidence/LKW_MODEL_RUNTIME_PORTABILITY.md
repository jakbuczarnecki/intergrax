# LKW Model Runtime Portability Proof

- schema: `lkw.model_runtime_portability.proof.v1`
- proof_id: `lkw-model-runtime-proof:2c54661f223a`
- classification: controlled local-provider live LKW product proof for exact Ollama and vLLM provider/model pairs
- overall: **PASS**
- repository_commit: `19daaf6934ae8a28bac855674616a2d210dc092c`

## Qualified provider pairs

### ollama
- configured_model: `qwen2.5:7b`
- resolved_model: `qwen2.5:7b`
- server_model: `qwen2.5:7b`
- adapter_class: `LangChainOllamaAdapter`
- server_version: `0.32.1`
- failure_code: `None`

### vllm
- configured_model: `Qwen/Qwen2.5-3B-Instruct`
- resolved_model: `Qwen/Qwen2.5-3B-Instruct`
- server_model: `Qwen/Qwen2.5-3B-Instruct`
- adapter_class: `VllmChatAdapter`
- server_version: `0.23.0`
- failure_code: `None`

## Shared index

- workspace_id: `76bcbe33-c6ed-4e97-9cdd-64b837b3ab66`
- source_id: `src:66b096c85c7474dfa8c369c88fe33b4ee7be9fb96092c4c9f6a44394087533c9`
- document_id: `lkwdoc:aa53acb5e7c1249d18db706e44e155b0`
- vector_count: `1`
- embedding_provider: `hf`
- embedding_model: `sentence-transformers/all-MiniLM-L6-v2`

## Limitations

- exact configured Ollama and vLLM pairs only; not universal model parity
- runtime hot swapping not required or proven