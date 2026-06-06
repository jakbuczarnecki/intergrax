# Intergrax infrastructure — host port matrix

All services bind to **localhost** unless noted. Use these ports in `INTERGRAX_*` env vars and integration tests.

| Service | Host port(s) | Profile | Integration slug / use |
|---------|--------------|---------|-------------------------|
| Redis | 6379 | core | `redis`, `celery` broker |
| PostgreSQL | 5432 | core | `postgresql`, Langfuse, Temporal DB |
| Kafka | 9092 | queue | `kafka` |
| RabbitMQ | 5672, 15672 (UI) | queue | `rabbitmq` |
| NATS | 4222, 8222 (monitor) | queue | `nats` |
| Qdrant | 6333, 6334 | rag | `qdrant` |
| Chroma | 8000 | rag | `chroma` |
| Weaviate | 8080, 50051 | rag | `weaviate` |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | rag | `neo4j`, GraphRAG |
| Milvus | 19530, 9091 | rag | `milvus` |
| Ollama | 11434 | rag | RAG/LLM embeddings |
| Docling | 8081 → container 8080 | rag | `docling` server mode |
| MongoDB | 27017 | data | `mongodb` |
| MySQL | 3306 | data | `mysql` |
| Cassandra | 9042 | data | `cassandra` |
| MinIO | 9000 (API), 9001 (console) | data | `minio`, S3 via endpoint URL |
| Memcached | 11211 | data | `memcached` |
| Vault | 8200 | secrets | `vault` |
| Elasticsearch | 9200 | observability | `elasticsearch`, `logs.search` |
| Prometheus | 9090 | observability | `prometheus`, `metrics.query_instant` |
| ClickHouse | 8123 (HTTP), **9002** (native, avoids MinIO 9000) | observability | `clickhouse` |
| Langfuse | 3000 | observability | `langfuse`, traces |
| Phoenix | 6006, 4317 | observability | `phoenix` |
| Mailpit | 1025 (SMTP), 8025 (UI) | observability | `email_smtp` tests |
| LocalStack | 4566 | cloud | `s3`, `sqs`, `dynamodb` |
| Azurite | 10000–10002 | cloud | `azure_blob` |
| Pub/Sub emulator | 8085 | cloud | `pubsub` |
| Temporal | 7233 | heavy | `temporal` |
| Vespa | **8089** (app, avoids Weaviate 8080), 19071 | heavy | `vespa` |
| Selenium | 4444, 7900 (VNC) | heavy | `selenium` |
| Keycloak | 8088 | heavy, p6 | `keycloak` identity provider |
| Typesense | 8108 | rag, p6 | `typesense` hybrid search |
| Airflow | 8086 → container 8080 | heavy, p6 | `airflow` workflow orchestrator |

## Resolved conflicts

| Conflict | Resolution |
|----------|------------|
| Chroma 8000 vs Weaviate | Weaviate **8080**, Chroma **8000** — different ports |
| MinIO 9000 vs ClickHouse native 9000 | ClickHouse native mapped to host **9002** |
| Weaviate 8080 vs Vespa default 8080 | Vespa app on host **8089** |
| Pub/Sub vs Weaviate | Emulator on **8085** |
| Prometheus 9090 vs Kafka | Kafka **9092** — no overlap |

## Compose profiles

| Profile | Services |
|---------|----------|
| `core` | redis, postgresql |
| `queue` | kafka, rabbitmq, nats |
| `rag` | qdrant, chroma, weaviate, neo4j, milvus, ollama, docling |
| `data` | mongodb, mysql, cassandra, minio, memcached |
| `secrets` | vault |
| `observability` | elasticsearch, prometheus, clickhouse, langfuse, phoenix, mailpit |
| `cloud` | localstack, azurite, pubsub-emulator |
| `heavy` | temporal, vespa, selenium, keycloak, airflow |
| `p6` | keycloak, typesense, airflow (M.6 P6 lab services) |
| `all` | alias — same as enabling every profile above |

**Default stack** (`./manage.sh start`): `core` + `queue` + `rag` + `data` + `secrets`.
