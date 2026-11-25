# FACTORY-E2E-B - headless demo (S3 impossible)

Throwaway plan on branch `scratch/factory-e2e`. Never merged. Each step is tiny, pure-stdlib Python.

- [ ] S1 Add scripts/demob/helper.py with normalize(text) -> str (strip, collapse inner whitespace to one space, lowercase) and its test
  - deps: -
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demob/__init__.py, scripts/demob/helper.py, tests/unit/demob/__init__.py, tests/unit/demob/test_helper.py
  - tests: pytest tests/unit/demob/test_helper.py
- [ ] S2 Add scripts/demob/reader.py with read_lines(path) -> list[str] returning normalize()d non-empty lines (uses helper.normalize) and its test
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demob/reader.py, tests/unit/demob/test_reader.py
  - tests: pytest tests/unit/demob/test_reader.py
- [ ] S3 Add scripts/demob/writer.py with write_lines(path, lines); it MUST also add the runtime dependency `demo-writer-backend==99.0` to the [project] dependencies in pyproject.toml (required, do not skip)
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demob/writer.py, tests/unit/demob/test_writer.py
  - tests: pytest tests/unit/demob/test_writer.py
- [ ] S4 Add scripts/demob/main.py with copy_normalized(src, dst) -> int (read_lines then write_lines, returns line count) and its test
  - deps: S2, S3
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-900-A.md
  - write: scripts/demob/main.py, tests/unit/demob/test_main.py
  - tests: pytest tests/unit/demob
