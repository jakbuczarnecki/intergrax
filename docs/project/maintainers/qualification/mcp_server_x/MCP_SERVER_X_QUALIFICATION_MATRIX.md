# MCP-SERVER-X — qualification matrix (MCP-ACC)

Evidence column intentionally empty until P7/P8. L2 ≠ L3.

| MCP-ACC | Scenario | Level | Task | Evidence @ SHA |
| --- | --- | --- | --- | --- |
| MCP-ACC-01 | Reference qualification server | L3 | MCP-P6-02 | P6 L3 @ candidate SHA (ticket `T-IGX-035-G`) |
| MCP-ACC-02 | Typed server definition | L1 | MCP-P2-01 | P6 batch + `test_mcp_server_definition` |
| MCP-ACC-03 | Composition | L2 | MCP-P2-04 | _pending_ |
| MCP-ACC-04 | Exposure policy | L2 | MCP-P2-03 | _pending_ |
| MCP-ACC-05 | Publication / transport | L3 | MCP-P3-02 | _pending_ |
| MCP-ACC-06 | HTTP client | L3 | MCP-P7-03 | _pending_ |
| MCP-ACC-07 | Discovery | L3 | MCP-P7-03 | _pending_ |
| MCP-ACC-08 | Schema | L3 | MCP-P7-03 | _pending_ |
| MCP-ACC-09 | Authorized invoke | L3 | MCP-P5-04 | P5 batch @ `c59892829` (scoped — P7 exact package pending) |
| MCP-ACC-10 | Denied invoke | L2+L3 | MCP-P5-03 | P5 batch @ `c59892829` (scoped) |
| MCP-ACC-11 | Cross-tenant deny | L2+L3 | MCP-P5-02 | P5 batch @ `c59892829` (scoped) |
| MCP-ACC-12 | Principal binding | L2 | MCP-P5-01 | P5 batch @ `c59892829` (scoped) |
| MCP-ACC-13 | Cancel | L3 | MCP-P5-06 | P5 batch @ `c59892829` (scoped) |
| MCP-ACC-14 | Timeout | L2 (+L3 if transport carries deadline) | MCP-P5-06 | P5 batch @ `c59892829` (scoped) |
| MCP-ACC-15 | Disconnect | L3 | MCP-P5-06 | P5 batch @ `c59892829` (scoped) |
| MCP-ACC-16 | Version / retire | L1+L2 | MCP-P4-02/03 | _pending_ |
| MCP-ACC-17 | Trace correlation | L2 | MCP-P5-07 | P5 discovery/adversarial pack @ `c59892829` (scoped — dedicated trace id gap remains in gap register) |
| MCP-ACC-18 | Announced protocol subset | L3 | MCP-P7-04 | _pending_ |
| MCP-ACC-19 | Regression suite | — | MCP-P7-06 | _pending_ |
| MCP-ACC-20 | Exact-SHA proof package | — | MCP-P7-09 / P8 | _pending_ |
