# Memory — Visual architecture (Mermaid)

**Canonical narrative:** [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md)  
**Audience:** maintainers verifying flows against code at `development` HEAD.

Labels: **CANONICAL** · **CONTRACT** · **PROJECTION** · **PROVIDER** · **VENDOR** · **COMPOSITION** · **AUTHORITY**

---

## 1. Layer architecture

```mermaid
flowchart TB
    subgraph composition [COMPOSITION Applications]
        APP[Applications / host wiring]
    end
    subgraph control [Memory Control / Services]
        CP[MemoryControlPlane]
        UPM[UserProfileManager]
    end
    subgraph contracts [CONTRACTS Lifecycle / Strategies]
        MC[memory_control.py]
        ML[memory_lifecycle.py]
        STR[strategies / recall pipeline]
    end
    subgraph providers [PROVIDER abstractions]
        UPS[UserProfileStore]
        STI[SessionTurnIndexStore]
        TMP[TaskMemoryPersistence]
    end
    subgraph integrations [Integrations / Vendors]
        INT[memory_wiring / vector ports]
        VEN[SQLite · Mongo DocumentStore · Qdrant · pgvector · Chroma]
    end

    APP --> CP
    CP --> UPM
    UPM --> MC
    UPM --> ML
    CP --> STR
    UPM --> UPS
    UPS --> INT --> VEN
    STI --> INT
    TMP --> INT
```

Upper layers depend on **contracts**; lower layers **implement** contracts. Memory core does not import vendor SDKs.

---

## 2. Authority map (canonical vs derived)

```mermaid
flowchart LR
    subgraph canonical [CANONICAL]
        UP[UserProfileStore / entries]
        TASK[TaskMemoryPersistence]
        ORG[OrganizationProfileStore]
    end
    subgraph derived [DERIVED / INDEXED]
        LTM[LTM vector projection]
        ENT[Entity temporal projection]
        STI[SessionTurnIndex]
        LH[Long-horizon compaction]
    end
    subgraph external [NOT MEMORY AUTHORITY]
        RAG[RAG evidence retrieval]
        CE[Context Engineering final context]
    end

    UP --> LTM
    UP --> ENT
    UP --> LH
    RAG --> CE
    UP --> CE
    STI --> CE
```

**SessionTurnIndex** is authoritative for episodic index reads in its domain; it is **not** USER Memory canonical authority.

---

## 3. USER canonical flow

```mermaid
flowchart TB
    APP[Application]
    CP[MemoryControlPlane]
    CAP[UserProfileMemoryCapability]
    MGR[UserProfileManager]
    CAN[(UserProfileStore CANONICAL)]
    PRJ[MemoryProjection PROJECTION]

    APP --> CP --> CAP --> MGR --> CAN
    MGR --> PRJ
```

---

## 4. USER remember (sequence)

```mermaid
sequenceDiagram
    participant App as Application
    participant CP as MemoryControlPlane
    participant Gov as Governance
    participant UPM as UserProfileManager
    participant Store as Canonical Store
    participant LC as LifecycleCoordinator
    participant Prj as Projection
    participant Diag as Diagnostics

    App->>CP: remember(identity, scope, request)
    CP->>CP: assert_memory_scope_authorized
    CP->>Gov: evaluate REMEMBER
    alt DENY
        Gov-->>CP: deny
        CP->>Diag: DENIED
    else permit
        CP->>UPM: remember
        UPM->>Store: primary write
        UPM->>LC: projection lifecycle
        LC->>Prj: upsert
        alt projection fail
            LC-->>UPM: PARTIAL_PROJECTION_FAILURE
            CP->>Diag: PARTIAL
        else success
            CP->>Diag: SUCCESS
        end
    end
```

---

## 5. USER recall (sequence)

```mermaid
sequenceDiagram
    participant App as Application
    participant CP as MemoryControlPlane
    participant UPM as UserProfileManager
    participant Pipe as Recall pipeline
    participant Gov as Governance disclosure
    participant Rank as Ranking / conflict strategies
    participant Diag as Diagnostics

    App->>CP: recall(identity, scope, request)
    CP->>CP: assert_memory_scope_authorized
    CP->>UPM: list / load candidates
    UPM->>Pipe: filter active canonical
    Pipe->>Gov: disclosure filter
    Pipe->>Rank: rank + conflict detect/resolve
    Rank-->>CP: MemoryControlRecallResult
    CP->>Diag: SUCCESS / DENIED / ...
```

---

## 6. USER forget (sequence)

```mermaid
sequenceDiagram
    participant CP as MemoryControlPlane
    participant Gov as Governance
    participant UPM as UserProfileManager
    participant Store as Canonical Store
    participant Prj as Projection
    participant Diag as Diagnostics

    CP->>CP: scope authorization
    CP->>Gov: FORGET
    CP->>UPM: forget
    UPM->>Store: canonical delete
    UPM->>Prj: projection delete
    alt projection delete fail
        UPM-->>CP: partial lifecycle
        CP->>Diag: PARTIAL
    else
        CP->>Diag: SUCCESS
    end
```

---

## 7. Supersession

```mermaid
flowchart LR
    OLD[old memory record]
    NEW[new memory record]
    OLD -->|superseded_by| NEW
    NEW -->|supersedes| OLD
    OLD -.->|no longer current| X[inactive]
    NEW -->|current authority| Y[active]
```

---

## 8. Projection failure and reconcile

```mermaid
flowchart TB
    CAN[canonical primary commit OK]
    PRJ[projection failure]
    PART[PARTIAL_PROJECTION_FAILURE]
    REC[reconcile USER scope]
    AUTH[authoritative active IDs from canonical]
    FIX[repair / remove orphans in projection]
    OUT[CONSISTENT or REPAIRED or FAILED]

    CAN --> PRJ --> PART --> REC
    REC --> AUTH --> FIX --> OUT
```

Reconciliation never flips authority to projection.

---

## 9. SESSION recall (sequence)

```mermaid
sequenceDiagram
    participant App as Application
    participant CP as MemoryControlPlane
    participant Epi as Episodic / STI capability
    participant Vec as VectorSessionTurnIndexStore
    participant VSM as VectorstoreManager
    participant Ven as VENDOR backend

    App->>CP: recall SESSION scope
    CP->>CP: assert_memory_scope_authorized
    CP->>Epi: session turn retrieval
    Epi->>Vec: query index
    Vec->>VSM: vector port
    VSM->>Ven: search
    Ven-->>App: episodic hits not USER canonical
```

---

## 10. TASK write / forget (sequence)

```mermaid
sequenceDiagram
    participant App as Application
    participant CP as MemoryControlPlane
    participant Task as TaskMemoryCapability
    participant Coord as TaskMemoryCoordinator
    participant Store as TaskMemoryPersistence

    App->>CP: remember TASK scope
    CP->>CP: scope authorization
    CP->>Task: remember_task KV
    Task->>Coord: persist
    Coord->>Store: durable write

    Note over App,Store: TASK recall via MemoryControlPlane unsupported use TaskMemoryCapability.read
```

---

## 11. Organization read / write

```mermaid
sequenceDiagram
    participant Nexus as Session / Nexus
    participant OPM as OrganizationProfileManager
    participant Store as OrganizationProfileStore
    participant SQLite as SQLiteOrganizationProfileStore

    Nexus->>Nexus: organization_id from session.tenant_id
    Nexus->>OPM: get / update profile
    OPM->>Store: read/write
    Store->>SQLite: SQL persistence
```

Organization authority is host-scoped (`organization_id`); store does not infer it from arbitrary payloads.

---

## 12. SessionTurnIndex vendor topology

```mermaid
flowchart TB
    STI[SessionTurnIndexStore CONTRACT]
    VST[VectorSessionTurnIndexStore]
    PORT[neutral vector ports]
    VSM[VectorstoreManager]
    Q[Qdrant VENDOR]
    PG[pgvector VENDOR]
    CH[Chroma HTTP persistent VENDOR]

    STI --> VST --> PORT --> VSM
    VSM --> Q
    VSM --> PG
    VSM --> CH
```

Chroma qual: HTTP persistent server, `IS_PERSISTENT=TRUE`, Docker volume, fresh `HttpClient` reconnect — **REAL_VENDOR_RECONNECT ≠ service restart proof**.

---

## 13. Task / Organization durable topology

```mermaid
flowchart TB
    TC[TaskMemoryCoordinator]
    TP[TaskMemoryPersistence]
    SQLITE_T[SQLiteTaskMemoryStore PROVIDER]

    OM[OrganizationProfileManager]
    OS[OrganizationProfileStore]
    SQLITE_O[SQLiteOrganizationProfileStore PROVIDER]

    TC --> TP --> SQLITE_T
    OM --> OS --> SQLITE_O
```

---

## 14. Provider qualification / admission

```mermaid
sequenceDiagram
    participant CFG as Configuration / MemoryProfile
    participant DIS as Discovery / classifier
    participant ID as Provider identity descriptor
    participant EV as Qualification evidence
    participant ADM as Admission evaluation
    participant MAT as Materialization
    participant HOST as Composition root

    CFG->>DIS: select plugin / integration
    DIS->>ID: provider_id + backing_provider_id
    ID->>EV: bind evidence
    EV->>ADM: fail-closed if PRODUCT requires DURABLE
    ADM->>MAT: materialize contract instance
    MAT->>HOST: wire MemoryControlPlane
```

---

## 15. Security path

```mermaid
flowchart LR
    RI[RequestIdentity AUTHORITY]
    SA[assert_memory_scope_authorized]
    GOV[MemorySecurityGovernanceService]
    OP[Memory operation]

    RI --> SA --> GOV --> OP
```

---

## 16. Observability (not authority)

```mermaid
flowchart LR
    OP[Memory operation]
    EM[MemoryDiagnosticEmitter]
    SK[MemoryObservabilitySink]

    OP --> EM --> SK
```

Outcomes include `SUCCESS`, `DENIED`, `FAILED`, `PARTIAL`, reconcile-related failures. Sink failure does not change Memory business outcomes.

---

## 17. Context Engineering integration

```mermaid
flowchart TB
    MEM[Memory recall outputs]
    RAG[RAG evidence]
    SES[Session context]
    CE[Context Engineering AUTHORITY]
    LLM[Model prompt]

    MEM --> CE
    RAG --> CE
    SES --> CE
    CE --> LLM
```

RAG retrieves documents; Memory retains canonical remembered state. Shared vector infrastructure ≠ shared authority.

---

## 18. Provider abstraction (VectorstoreManager)

```mermaid
flowchart TB
    SEM[Memory semantic STI layer]
    VSM[VectorstoreManager one abstraction]
    V1[Qdrant]
    V2[pgvector]
    V3[Chroma]

    SEM --> VSM
    VSM --> V1
    VSM --> V2
    VSM --> V3
```

---

## Diagram count

18 Mermaid diagrams (layer, authority, flows, topology, admission, CE, observability, security). Validate syntax when editing; no screenshots.
