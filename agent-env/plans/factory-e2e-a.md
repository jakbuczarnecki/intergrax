# FACTORY-E2E-A - headless demo (all steps feasible)

Throwaway plan on branch `scratch/factory-e2e`. Never merged. Each step is tiny, pure-stdlib Python.

- [ ] S1 Add scripts/demo/helper.py with normalize(text) -> str (strip, collapse inner whitespace to one space, lowercase) and its test
  - deps: -
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demo/__init__.py, scripts/demo/helper.py, tests/unit/demo/__init__.py, tests/unit/demo/test_helper.py
  - tests: pytest tests/unit/demo/test_helper.py
- [ ] S2 Add scripts/demo/reader.py with read_lines(path) -> list[str] returning normalize()d non-empty lines (uses helper.normalize) and its test
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demo/reader.py, tests/unit/demo/test_reader.py
  - tests: pytest tests/unit/demo/test_reader.py
- [ ] S3 Add scripts/demo/writer.py with write_lines(path, lines) writing one line each with a trailing newline, UTF-8, and its test
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demo/writer.py, tests/unit/demo/test_writer.py
  - tests: pytest tests/unit/demo/test_writer.py
- [ ] S4 Add scripts/demo/main.py with copy_normalized(src, dst) -> int (read_lines then write_lines, returns line count) and its test
  - deps: S2, S3
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demo/main.py, tests/unit/demo/test_main.py
  - tests: pytest tests/unit/demo
