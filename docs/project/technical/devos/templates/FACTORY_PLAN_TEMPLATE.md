# FACTORY-DEMO - minimal factory plan

Replace `T-IGX-NNN-A`, the write sets and the tests with real values. The plan id is the file name without
`.md`. Every ticket must exist under `governance/tickets/` and contain an `Approved:` line, otherwise
`igx_factory.py start` refuses (preflight). Parallel steps (S2, S3) need disjoint write sets.

- [ ] S1 Add the shared helper
  - deps: -
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-NNN-A.md
  - write: scripts/demo/helper.py, tests/unit/demo/test_helper.py
  - tests: pytest tests/unit/demo/test_helper.py
- [ ] S2 Add the reader (parallel with S3)
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-NNN-A.md
  - write: scripts/demo/reader.py, tests/unit/demo/test_reader.py
  - tests: pytest tests/unit/demo/test_reader.py
- [ ] S3 Add the writer (parallel with S2)
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-NNN-A.md
  - write: scripts/demo/writer.py, tests/unit/demo/test_writer.py
  - tests: pytest tests/unit/demo/test_writer.py
- [ ] S4 Wire reader and writer together
  - deps: S2, S3
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-NNN-A.md
  - write: scripts/demo/main.py, tests/unit/demo/test_main.py
  - tests: pytest tests/unit/demo
