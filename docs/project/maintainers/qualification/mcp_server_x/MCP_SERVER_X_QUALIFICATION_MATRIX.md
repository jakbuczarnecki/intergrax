# MCP-SERVER-X — qualification matrix (MCP-ACC)

P7 evidence below is pinned to `TEST_EXECUTION_SHA` `abbe1525450da77da6801b080619a600c20cd497` ([package](./MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md)); L2 ≠ L3. Rows marked scoped carry the limitation named in the package.

| MCP-ACC | Scenario | Level | Task | Evidence @ SHA |
| --- | --- | --- | --- | --- |
| MCP-ACC-01 | Reference qualification server | L3 | MCP-P6-02 | P7 `COVERED`: 3 L3 tests @ `abbe15254` |
| MCP-ACC-02 | Typed server definition | L1 | MCP-P2-01 | P7 `COVERED`: 17 L1 tests (+ P6 batch, `test_mcp_server_definition`) @ `abbe15254` |
| MCP-ACC-03 | Composition | L2 | MCP-P2-04 | P7 `COVERED`: 10 L2 tests @ `abbe15254` |
| MCP-ACC-04 | Exposure policy | L2 | MCP-P2-03 | P7 `COVERED`: 8 L1 + 8 L2 tests @ `abbe15254` |
| MCP-ACC-05 | Publication / transport | L3 | MCP-P3-02 | P7 `COVERED`: 3 L3 tests @ `abbe15254` |
| MCP-ACC-06 | HTTP client | L3 | MCP-P7-03 | P7 `COVERED`: 2 L3 tests @ `abbe15254` |
| MCP-ACC-07 | Discovery | L3 | MCP-P7-03 | P7 `COVERED`: 4 L3 tests @ `abbe15254` |
| MCP-ACC-08 | Schema | L3 | MCP-P7-03 | P7 `COVERED`: 5 L3 tests @ `abbe15254` |
| MCP-ACC-09 | Authorized invoke | L3 | MCP-P5-04 | P7 `COVERED — scoped`: 3 L3 tests (`run_agent` path) + P5 batch @ `c59892829` @ `abbe15254` |
| MCP-ACC-10 | Denied invoke | L2+L3 | MCP-P5-03 | P7 `COVERED`: 10 L2 + 5 L3 tests (+ P5 batch) @ `abbe15254` |
| MCP-ACC-11 | Cross-tenant deny | L2+L3 | MCP-P5-02 | P7 `COVERED`: 5 L2 + 3 L3 tests (+ P5 batch) @ `abbe15254` |
| MCP-ACC-12 | Principal binding | L2 | MCP-P5-01 | P7 `COVERED`: 8 L2 tests (+ P5 batch) @ `abbe15254` |
| MCP-ACC-13 | Cancel | L3 | MCP-P5-06 | P7 `COVERED`: 4 L2 + 2 L3 tests (+ P5 batch); see R-P7-04 @ `abbe15254` |
| MCP-ACC-14 | Timeout | L2 (+L3 if transport carries deadline) | MCP-P5-06 | P7 `COVERED — scoped`: 3 L2 + 3 L3 tests; canonical-execution deadline N/A through MCP (R-P7-04) @ `abbe15254` |
| MCP-ACC-15 | Disconnect | L3 | MCP-P5-06 | P7 `COVERED`: 1 L2 supplement + 2 L3 tests (+ P5 batch) @ `abbe15254` |
| MCP-ACC-16 | Version / retire | L1+L2 | MCP-P4-02/03 | P7 `COVERED`: 7 L1 + 6 L2 tests @ `abbe15254` |
| MCP-ACC-17 | Trace correlation | L2 | MCP-P5-07 | P7 `COVERED — scoped`: 4 L2 tests; no dedicated trace id (R-P7-01, DR-1) @ `abbe15254` |
| MCP-ACC-18 | Announced protocol subset | L3 | MCP-P7-04 | P7 `COVERED — scoped`: 10 L3 tests; Resources/Prompts advertised but empty @ `abbe15254` |
| MCP-ACC-19 | Regression suite | — | MCP-P7-06 | P7 `COVERED`: regression selection 141 passed; 11 pre-existing failures excluded (R-P7-02) @ `abbe15254` |
| MCP-ACC-20 | Exact-SHA proof package | — | MCP-P7-09 / P8 | P7 `COVERED`: exact-SHA package record @ `abbe15254` |
