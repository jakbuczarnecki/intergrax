# Intergrax Infrastructure Layer

This directory contains infrastructure modules used by the Intergrax platform.

Each infrastructure component is isolated and self-contained.

Examples of infrastructure components:

* Redis
* PostgreSQL
* Kafka
* RabbitMQ
* Qdrant
* ChromaDB
* Docling

Infrastructure services are **fully decoupled from the Intergrax runtime code** (`intergrax/`).

They are used for:

* local development
* CI pipelines
* integration tests
* production infrastructure services

---

# Directory Structure

```
infra/
  docker/

    redis/
      docker-compose.yml

    postgresql/
      docker-compose.yml

    kafka/
      docker-compose.yml

    qdrant/
      docker-compose.yml

    chromadb/
      docker-compose.yml

    docling/
      Dockerfile
      docker-compose.yml
      service.py

    manage.ps1
    manage.sh
```

Each tool directory contains its own Docker configuration.

Possible files inside a tool directory:

| File               | Description                     |
| ------------------ | ------------------------------- |
| docker-compose.yml | Container configuration         |
| Dockerfile         | Optional image build definition |
| service files      | Optional service implementation |

---

# Infrastructure Management

Infrastructure lifecycle is managed using two scripts:

```
manage.ps1
manage.sh
```

* **manage.ps1** — Windows
* **manage.sh** — Linux / macOS / CI

Both scripts provide the same functionality.

Supported actions:

```
build
start
stop
status
```

---

# Build Docker Images

Images are built **only for tools that contain a Dockerfile**.

If a tool does not provide a Dockerfile, it will be skipped.

## Build a single tool

Linux / macOS:

```
./manage.sh docling build
```

Windows:

```
.\manage.ps1 docling build
```

---

## Build all tools

Linux / macOS:

```
./manage.sh all build
```

Windows:

```
.\manage.ps1 all build
```

This command scans all tool directories and builds images only for those containing a Dockerfile.

---

# Start a Service

Start a single infrastructure service.

Linux / macOS:

```
./manage.sh redis start
```

Windows:

```
.\manage.ps1 redis start
```

Example:

```
./manage.sh docling start
```

---

# Stop a Service

Linux / macOS:

```
./manage.sh redis stop
```

Windows:

```
.\manage.ps1 redis stop
```

---

# Check Service Status

Linux / macOS:

```
./manage.sh redis status
```

Windows:

```
.\manage.ps1 redis status
```

This command shows container status using:

```
docker compose ps
```

---

# Example Workflow

Typical development workflow:

### 1. Build image

```
./manage.sh docling build
```

### 2. Start service

```
./manage.sh docling start
```

### 3. Check status

```
./manage.sh docling status
```

### 4. Stop service

```
./manage.sh docling stop
```

---

# CI/CD Usage

Infrastructure services can be controlled directly in CI pipelines.

Example:

```
./infra/docker/manage.sh docling build
./infra/docker/manage.sh docling start
```

This allows CI to:

* build images
* start infrastructure services
* run integration tests
* shut down services after tests

---

# Adding a New Infrastructure Service

To add a new infrastructure tool:

### 1. Create directory

```
infra/docker/<tool>/
```

Example:

```
infra/docker/vector-db/
```

### 2. Add docker-compose.yml

```
docker-compose.yml
```

### 3. (Optional) Add Dockerfile

```
Dockerfile
```

If a Dockerfile exists, the image can be built using:

```
manage build
```

---

# Example: Docling Service

Location:

```
infra/docker/docling
```

Docling provides a document parsing service used by the Intergrax RAG ingestion pipeline.

Capabilities:

* PDF parsing
* OCR fallback
* structured document extraction
* markdown export

Build the image:

```
./manage.sh docling build
```

Start the service:

```
./manage.sh docling start
```

Health endpoint:

```
http://localhost:8081/health
```

Parse endpoint:

```
http://localhost:8081/parse
```

---

# Design Principles

The infrastructure layer follows several architectural rules.

### Decoupled from runtime

Infrastructure services are isolated from the Intergrax runtime code.

### Deterministic lifecycle

All services are controlled using a single interface.

### Tool isolation

Each infrastructure tool lives in its own directory.

### Optional build

Images are built only when a Dockerfile is present.

---

# Summary

Infrastructure services are managed using:

```
manage.ps1
manage.sh
```

Supported operations:

```
build
start
stop
status
```

This ensures consistent infrastructure management across:

* local development
* CI pipelines
* integration environments
* production deployments
