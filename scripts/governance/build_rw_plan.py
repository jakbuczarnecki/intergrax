#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Build canonical T-AF2-RW-PLAN artefacts from the FRZ matrix, approved ticket plans and reviewed bindings.

Inputs: the FRZ acceptance matrix, ``rw_plan_bindings.v1.json`` (reviewed data), the ticket
plans named by the bindings and, for red-first, the prior plan and the red-first record.
The generator fails closed: a missing write-set section, an unordered overlapping ticket
pair, an invalid lane, an unmapped OPEN row or a class marker used as a negative test stops it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MATRIX = _REPO / "docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md"
_TICKETS = _REPO / "governance/tickets"
_PLANS = _TICKETS / "plans"
_AF = _REPO / "docs/project/maintainers/qualification/arch_freeze_program"
_RW_PLAN_MD = _AF / "RW_PLAN.md"
_RW_PLAN_JSON = _AF / "rw_plan.v1.json"
_BINDINGS_JSON = _AF / "rw_plan_bindings.v1.json"

APPENDIX_A_MARK = "## Appendix A."
APPENDIX_END_MARKERS = ("## Appendix B", "## Appendix C", "## Appendix D")
APPENDIX_C_MARK = "## Appendix C."
_RW_TICKET_RE = re.compile(r"RW ticket (T-IGX-\d{3}-[A-Z0-9]+)")
_NT_RE = re.compile(r"(NT-[A-Z0-9]+-\d+|NT-[A-Z]{3}-\d+|NT-ROW-\d+)")
_FRZ_ROW_RE = re.compile(r"^\|\s*`(FRZ-[A-Z]+-\d+)`\s*\|")
_FINDING_RE = re.compile(r"\bA\d-\d{3}\b")
_NODE_RE = re.compile(
    r"\b(?:PT-[A-Z0-9]+(?:-[A-Z0-9]+)*|T-AF2-[A-Z0-9]+(?:-[A-Z0-9]+)*|T-B7-[A-Z0-9]+(?:-[A-Z0-9]+)*"
    r"|SSOT-[A-Z0-9]+(?:-[A-Z0-9]+)*|T-IGX-\d{3}-[A-Z0-9]+)"
)
APPENDIX_A_COL = {
    "frz_id": 1,
    "summary": 2,
    "semantic_owner": 3,
    "matrix_status": 7,
    "tests": 10,
    "negative_tests": 11,
    "required_fix": 12,
    "required_certification": 13,
    "pass_condition": 15,
}
_REQUIRED_COLUMNS = ("frz_id", "summary", "tests", "negative_tests", "required_fix", "required_certification")

CLASS_MARKERS = ("NEG-0", "NEG-P")
NEGATIVE_TEST_ENTRY_KEYS = ("nt_id", "node", "mechanism", "owner_ticket")
LANE_CAP = 5
ACTIVE_WRITER_LIMIT = 2
PT_WA_TICKET = "T-IGX-043-G"
FIND_AF2_IDS = tuple(f"FIND-AF2-{n:02d}" for n in range(1, 24))

AUDIT_FLAG_CONFLICTS = [
    {"finding_id": "A4-005", "owner_ticket": "T-IGX-051-H", "disposition": "T-AF2-DEPS-BATCH; PT-WF overlap"},
    {
        "finding_id": "A4-018",
        "owner_ticket": "T-IGX-051-D",
        "disposition": "T-AF2-RW-RUNTIME; shares PT-WF / FRZ-CTR rows",
    },
    {"finding_id": "A6-005", "owner_ticket": "T-IGX-043-I", "disposition": "PT-WD; ENT manifest engine scope"},
    {
        "finding_id": "A4-006",
        "owner_ticket": "T-IGX-078-D",
        "owner_node": "PT-T01",
        "owner_wave": "W3",
        "owner_source": "WP12 DAG / PT-T01 / A4-006; GA.2R T-IGX-078-D",
        "disposition": "PT-T01 gate emitter / tenant propagation",
    },
    {"finding_id": "A3-018", "owner_ticket": "T-IGX-046-G", "disposition": "PT-G02 STATE-X citation repair"},
    {"finding_id": "A4-016", "owner_ticket": "T-IGX-043-H", "disposition": "PT-WB dual host LLM resolver"},
]

SCOPE_HARD_PREDECESSORS = [
    "M1",
    "DS-1",
    "DS-2",
    "DS-3",
    "DS-4",
    "DS-EXC",
    "PT-V04",
    "PT-B01",
    "PT-B02",
    "REV-MASTER",
    "T-AF2-RW-PLAN",
    "T-AF2-RW-PLAN-AUDIT",
]

_TYP_SOURCE = "MEP-AF2 v2.1 / section G / DP-36; GA.2R re-ownership (T-IGX-078-C, WL-K1-B-02)"
_TEN_SOURCE = "WP12 DAG / PT-T01 / A4-006; GA.2R T-IGX-078-D"


def _ga2r_contract(ticket: str, node: str, source: str) -> dict[str, object]:
    return {
        "owner_ticket": ticket,
        "owner_node": node,
        "owner_wave": "W3",
        "owner_source": source,
        "owner_provenance": "explicit_ga2r",
    }


# Source-backed CODE_FIX ownership (MEP-AF2 v2.1 / WP12 DAG, GA.2R re-ownership); exact FRZ ids only.
EXPLICIT_CODE_FIX_OWNERSHIP: dict[str, dict[str, object]] = {
    "FRZ-TYP-01": _ga2r_contract("T-IGX-078-C", "T-AF2-RW-TYP", _TYP_SOURCE),
    "FRZ-TYP-03": _ga2r_contract("T-IGX-078-C", "T-AF2-RW-TYP", _TYP_SOURCE),
    "FRZ-RPL-04": _ga2r_contract("T-IGX-078-C", "T-AF2-RW-TYP", _TYP_SOURCE),
    "FRZ-TEN-02": _ga2r_contract("T-IGX-078-D", "PT-T01", _TEN_SOURCE),
    "FRZ-TEN-07": _ga2r_contract("T-IGX-078-D", "PT-T01", _TEN_SOURCE),
    "FRZ-FRZ-10": {
        "owner_ticket": None,
        "owner_node": "T-B7-PFP-ACTIVATE",
        "owner_wave": "W5",
        "verification_node": "T-B7-PFP-VERIFY",
        "verification_wave": "W8",
        "owner_source": "MEP-AF2 v2.1 / W5 and W8",
        "owner_ticket_family": "T-IGX-052-*",
        "owner_provenance": "explicit_mep_wp12",
    },
}


@dataclass
class CodeFixRow:
    frz_id: str
    summary: str
    negative_test: str | None
    remediation_track: str
    owner_ticket: str | None
    regression_scope: str | None
    delta_findings: list[str] = field(default_factory=list)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_REPO, text=True, encoding="utf-8")


def _git_head() -> str:
    return _git("rev-parse", "HEAD").strip()


def _pin_tree(pin: str) -> frozenset[str]:
    return frozenset(_git("ls-tree", "-r", "--name-only", pin).splitlines())


def _pin_text(pin: str, path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"{pin}:{path}"], cwd=_REPO, text=True, encoding="utf-8", stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def load_bindings() -> dict:
    data = json.loads(_BINDINGS_JSON.read_text(encoding="utf-8"))
    if data.get("schema") != "rw_plan_bindings.v1":
        raise SystemExit("bindings schema must be rw_plan_bindings.v1")
    pin = data.get("read_pin")
    if not isinstance(pin, str) or len(pin) != 40:
        raise SystemExit("bindings read_pin must be a full commit sha")
    return data


# --- matrix -----------------------------------------------------------------


def _extract_appendix_a_section() -> str:
    text = _MATRIX.read_text(encoding="utf-8")
    start = text.find(APPENDIX_A_MARK)
    if start < 0:
        raise SystemExit(f"missing {APPENDIX_A_MARK} in matrix")
    section = text[start:]
    for marker in APPENDIX_END_MARKERS:
        end = section.find(marker)
        if end > 0:
            return section[:end]
    return section


def _row_parts(line: str) -> list[str]:
    return [p.strip() for p in line.split("|")]


def _column(parts: list[str], name: str) -> str:
    index = APPENDIX_A_COL[name]
    return parts[index] if index < len(parts) else ""


def parse_appendix_a_all_rows(section: str | None = None) -> list[dict[str, str | None]]:
    """Every Appendix A row with the columns the plan uses."""
    if section is None:
        section = _extract_appendix_a_section()
    rows: list[dict[str, str | None]] = []
    for line in section.splitlines():
        if not _FRZ_ROW_RE.match(line):
            continue
        parts = _row_parts(line)
        if len(parts) <= max(APPENDIX_A_COL[c] for c in _REQUIRED_COLUMNS):
            raise SystemExit(f"malformed Appendix A row (columns): {line[:120]}")
        row: dict[str, str | None] = {name: _column(parts, name) for name in APPENDIX_A_COL}
        row["frz_id"] = str(row["frz_id"]).strip("`")
        accountable = re.search(r"accountable:\s*(.+)$", str(row["semantic_owner"]))
        row["accountable"] = accountable.group(1).strip() if accountable else None
        rows.append(row)
    return rows


def parse_appendix_c(text: str | None = None) -> dict[str, dict[str, str]]:
    """Appendix C blocks: FRZ id to its labelled bullets (label lower-cased)."""
    if text is None:
        text = _MATRIX.read_text(encoding="utf-8")
    start = text.find(APPENDIX_C_MARK)
    if start < 0:
        return {}
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text[start + len(APPENDIX_C_MARK) :].splitlines():
        if line.startswith("## "):
            break
        head = re.match(r"^####\s+`(FRZ-[A-Z]+-\d+)`", line)
        if head:
            current = blocks.setdefault(head.group(1), {})
            continue
        bullet = re.match(r"^-\s+\*\*(.+?):\*\*\s*(.*)$", line)
        if current is not None and bullet:
            current[bullet.group(1).strip().lower()] = bullet.group(2).strip()
    return blocks


def _extract_negative_test(neg_col: str) -> str | None:
    nt_m = _NT_RE.search(neg_col)
    if nt_m:
        return nt_m.group(1)
    for marker in CLASS_MARKERS:
        if marker in neg_col:
            return marker
    return None


def _extract_owner_ticket(cert_col: str, fix_col: str) -> str | None:
    rw_m = _RW_TICKET_RE.search(cert_col)
    if rw_m:
        return rw_m.group(1)
    for pattern in (
        r"Appendix C,?\s*(T-IGX-\d{3}-[A-Z0-9]+)",
        r";\s*(T-IGX-\d{3}-[A-Z0-9]+)\)",
        r";\s*(T-IGX-\d{3}-[A-Z0-9]+)\s*\+",
    ):
        m = re.search(pattern, fix_col)
        if m and "*" not in m.group(1):
            return m.group(1)
    return None


def apply_explicit_ownership(frz_id: str, row: CodeFixRow) -> dict[str, object]:
    """Merge matrix parse with canonical explicit ownership for the listed FRZ rows."""
    explicit = EXPLICIT_CODE_FIX_OWNERSHIP.get(frz_id)
    if explicit is None:
        return {"owner_ticket": row.owner_ticket}
    out = dict(explicit)
    if row.owner_ticket and row.owner_ticket != out.get("owner_ticket"):
        out["matrix_owner_ticket_parse"] = row.owner_ticket
    return out


def _code_fix_row_key(row: CodeFixRow) -> tuple:
    return (
        row.summary,
        row.negative_test,
        row.remediation_track,
        row.owner_ticket,
        row.regression_scope,
        tuple(row.delta_findings),
    )


def parse_appendix_a_code_fix_inventory(section: str | None = None) -> list[CodeFixRow]:
    """Parse CODE_FIX rows from Appendix A required_fix column only."""
    seen: dict[str, CodeFixRow] = {}
    for cells in parse_appendix_a_all_rows(section):
        required_fix = str(cells["required_fix"])
        if not required_fix.startswith("CODE_FIX"):
            continue
        frz_id = str(cells["frz_id"])
        findings: set[str] = set()
        for name in ("semantic_owner", "required_fix", "pass_condition"):
            findings.update(_FINDING_RE.findall(str(cells[name])))
        regression: str | None = str(cells["tests"]).strip()
        if not regression or regression in {"—", "-", "none"}:
            regression = None
        row = CodeFixRow(
            frz_id=frz_id,
            summary=str(cells["summary"])[:500],
            negative_test=_extract_negative_test(str(cells["negative_tests"])),
            remediation_track=required_fix,
            owner_ticket=_extract_owner_ticket(str(cells["required_certification"]), required_fix),
            regression_scope=regression,
            delta_findings=sorted(findings),
        )
        if frz_id in seen and _code_fix_row_key(seen[frz_id]) != _code_fix_row_key(row):
            raise SystemExit(f"conflicting CODE_FIX records for {frz_id}")
        seen[frz_id] = row
    return list(seen.values())


def build_coverage(
    all_rows: list[dict[str, str | None]],
    inventory_ids: set[str],
    overrides: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Every OPEN matrix row outside the CODE_FIX inventory, with a reason and an owning node."""
    coverage: list[dict[str, object]] = []
    for row in all_rows:
        frz = str(row["frz_id"])
        if row["matrix_status"] != "OPEN" or frz in inventory_ids:
            continue
        certification = str(row["required_certification"])
        entry: dict[str, object] = {
            "frz_id": frz,
            "reason": str(row["required_fix"]),
            "owner_nodes": list(dict.fromkeys(_NODE_RE.findall(certification))),
            "accountable": row["accountable"],
            "certification": certification,
        }
        entry.update(overrides.get(frz, {}))
        if not entry["reason"] or not entry["owner_nodes"]:
            raise SystemExit(f"coverage: {frz} has no reason or no owning node")
        coverage.append(entry)
    stale = sorted(set(overrides) - {str(e["frz_id"]) for e in coverage})
    if stale:
        raise SystemExit(f"coverage overrides for rows that are not OPEN outside the inventory: {stale}")
    return coverage


def find_af2_errors(entries: object) -> list[str]:
    if not isinstance(entries, list):
        return ["FIND-AF2 mapping missing"]
    ids = [str(e.get("id")) for e in entries if isinstance(e, dict)]
    errors = [f"FIND-AF2 mapping missing {i}" for i in FIND_AF2_IDS if i not in ids]
    errors.extend(
        f"FIND-AF2 mapping unknown or duplicate {i}" for i in ids if i not in FIND_AF2_IDS or ids.count(i) > 1
    )
    errors.extend(f"FIND-AF2 {e.get('id')} has no mapping" for e in entries if not e.get("mapping"))
    return errors


# --- write sets ---------------------------------------------------------------

_PATH_ROOTS = r"(?:\.github|intergrax|agents|applications|scripts|tests|testing_support|governance|docs)"
_PATH_RE = re.compile(
    rf"(?<![\w/.\-])(?:{_PATH_ROOTS}/[\w./\-*]*[\w*/]|pyproject\.toml|uv\.lock)(?![\w\-/*])(?!\.\w)(?!@)(?!:\d)"
)
_EXCLUDED_HEADING_RE = re.compile(
    r"(?i)\b(?:forbidden|out of (?:the )?write set|out of scope|not in (?:this|the) write set)"
)
_EXCLUDED_PARAGRAPH_RE = re.compile(
    r"(?i)^(?:forbidden|out of (?:the )?write set|not in (?:this|the) write set|explicitly out|excluded)\b"
)
_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
_NOTE_START_RE = re.compile(r"\(|\s[—–]\s|\s-\s|#|←")
_HEADING_RE = re.compile(r"^(#{1,6})\s")


def _heading_matches(line: str, heading: str) -> bool:
    line = line.rstrip()
    return line == heading or line.startswith(heading + " ")


def _headings(lines: list[str]) -> list[tuple[int, int]]:
    """(line index, level) of headings outside fenced blocks."""
    found: list[tuple[int, int]] = []
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = _HEADING_RE.match(line)
        if match and not fenced:
            found.append((index, len(match.group(1))))
    return found


def _section_lines(lines: list[str], heading: str, occurrence: int | None) -> list[str]:
    headings = _headings(lines)
    matches = [(i, level) for i, level in headings if _heading_matches(lines[i], heading)]
    if not matches:
        raise SystemExit(f"write set: heading not found: {heading}")
    if occurrence is None and len(matches) > 1:
        raise SystemExit(f"write set: heading is ambiguous without occurrence: {heading}")
    if occurrence is not None and not 1 <= occurrence <= len(matches):
        raise SystemExit(f"write set: heading occurrence {occurrence} not found: {heading}")
    start, level = matches[(occurrence or 1) - 1]
    end = next((i for i, other in headings if i > start and other <= level), len(lines))
    return lines[start + 1 : end]


def _drop_excluded_subsections(lines: list[str], exclude_headings: list[str]) -> list[str]:
    headings = dict(_headings(lines))
    kept: list[str] = []
    skip_level: int | None = None
    for index, line in enumerate(lines):
        level = headings.get(index)
        if level is not None:
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and (
                _EXCLUDED_HEADING_RE.search(line) or any(_heading_matches(line, h) for h in exclude_headings)
            ):
                skip_level = level
            if skip_level is None:
                kept.append("")
                continue
        if skip_level is None:
            kept.append(line)
    return kept


def _blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] = []
    fence: list[str] | None = None
    for line in lines:
        if line.lstrip().startswith("```"):
            if fence is None:
                if current:
                    blocks.append(("text", current))
                    current = []
                fence = []
            else:
                blocks.append(("fence", fence))
                fence = None
            continue
        if fence is not None:
            fence.append(line)
        elif line.strip():
            current.append(line)
        elif current:
            blocks.append(("text", current))
            current = []
    if current:
        blocks.append(("text", current))
    return blocks


_BRACE_RE = re.compile(r"(?<![\w.\-/])([\w.\-/]*/[\w.\-/]*)\{([\w.\-/ ]+(?:,[\w.\-/ ]+)+)\}([\w.\-/]*)")


def _expand_braces(text: str) -> str:
    """Expand `dir/{a,b}.py` shorthand so every listed path is counted."""

    def _join(match: re.Match[str]) -> str:
        head, items, tail = match.group(1), match.group(2), match.group(3)
        return " ".join(f"{head}{item.strip()}{tail}" for item in items.split(",") if item.strip())

    return _BRACE_RE.sub(_join, text)


def _listed(text: str) -> list[str]:
    text = _expand_braces(text)
    note = _NOTE_START_RE.search(text)
    return _PATH_RE.findall(text[: note.start()] if note else text)


def _collect_text_block(lines: list[str], paths: set[str], prose: set[str]) -> None:
    in_item = False
    item_closed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            in_item = False
            for cell in stripped.strip("|").split("|"):
                if _PATH_RE.search(cell):
                    paths.update(_listed(cell.strip()))
                    break
            continue
        marker = _ITEM_RE.match(line)
        if marker:
            in_item = True
            body = line[marker.end() :]
            paths.update(_listed(body))
            item_closed = _NOTE_START_RE.search(body) is not None
        elif in_item and line[:1].isspace():
            if not item_closed:
                paths.update(_listed(stripped))
                item_closed = _NOTE_START_RE.search(stripped) is not None
        elif _HEADING_RE.match(line):
            in_item = False
        else:
            in_item = False
            prose.update(_PATH_RE.findall(line))


def extract_write_set(plan_text: str, spec: dict) -> dict[str, list[str]]:
    """Paths listed (fenced, table or bullet form) in the named write-set sections. No cap."""
    lines = plan_text.splitlines()
    paths: set[str] = set()
    prose: set[str] = set()
    for section in spec.get("sections", []):
        body = _section_lines(lines, section["heading"], section.get("occurrence"))
        body = _drop_excluded_subsections(body, list(spec.get("exclude_headings", [])))
        skip_next = False
        for kind, block in _blocks(body):
            if kind == "text":
                first = block[0].strip().lstrip("*_> ")
                if _EXCLUDED_PARAGRAPH_RE.match(first):
                    skip_next = block[-1].rstrip().rstrip("*_").endswith(":")
                    continue
            if skip_next:
                skip_next = False
                continue
            if kind == "fence":
                for line in block:
                    paths.update(_listed(_ITEM_RE.sub("", line)))
            else:
                _collect_text_block(block, paths, prose)
    for rewrite in spec.get("rewrites", []):
        if rewrite["from"] not in paths:
            raise SystemExit(f"write set: rewrite source not extracted: {rewrite['from']}")
        paths.remove(rewrite["from"])
        paths.add(rewrite["to"])
    for excluded in spec.get("exclude_paths", []):
        if excluded["path"] not in paths:
            raise SystemExit(f"write set: excluded path not extracted: {excluded['path']}")
        paths.remove(excluded["path"])
    paths.update(extra["path"] for extra in spec.get("extra_paths", []))
    if not paths:
        raise SystemExit(f"write set: no paths extracted from {spec.get('sections')}")
    expected = spec.get("expected_count")
    if expected is not None and len(paths) != expected:
        raise SystemExit(f"write set: expected {expected} paths, extracted {len(paths)}")
    return {"paths": sorted(paths), "prose_mentions": sorted(prose - paths)}


def label_path(path: str, tree: frozenset[str]) -> str:
    if "*" in path:
        return "glob"
    prefix = path if path.endswith("/") else path + "/"
    if path in tree or any(item.startswith(prefix) for item in tree):
        return "exists"
    return "new"


def _pattern_kind(path: str) -> tuple[str, str]:
    if path.endswith(("/*", "/**")):
        return "dir", path[: path.rindex("/") + 1]
    if path.endswith("/"):
        return "dir", path
    return ("glob", path) if "*" in path else ("file", path)


def paths_overlap(a: str, b: str) -> bool:
    if a == b:
        return True
    (kind_a, value_a), (kind_b, value_b) = sorted((_pattern_kind(a), _pattern_kind(b)))
    # sorted order of kinds: dir < file < glob
    if kind_a == "dir":
        if kind_b == "file":
            return value_b.startswith(value_a)
        literal = value_b.split("*", 1)[0]
        return literal.startswith(value_a) or value_a.startswith(literal)
    if kind_a == "file":
        if kind_b == "glob":
            return fnmatch.fnmatchcase(value_a, value_b)
        return value_a.startswith(value_b + "/") or value_b.startswith(value_a + "/")
    literal_a, literal_b = value_a.split("*", 1)[0], value_b.split("*", 1)[0]
    return literal_a.startswith(literal_b) or literal_b.startswith(literal_a)


def overlap_matrix(write_sets: dict[str, list[str]]) -> list[dict[str, object]]:
    names = sorted(write_sets)
    matrix: list[dict[str, object]] = []
    for position, a in enumerate(names):
        for b in names[position + 1 :]:
            shared: set[str] = set()
            for path_a in write_sets[a]:
                for path_b in write_sets[b]:
                    if paths_overlap(path_a, path_b):
                        shared.update((path_a, path_b))
            if shared:
                matrix.append({"a": a, "b": b, "paths": sorted(shared)})
    return matrix


# --- ordering and lanes --------------------------------------------------------


def _reachable(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    direct: dict[str, set[str]] = {}
    for first, second in edges:
        direct.setdefault(first, set()).add(second)
    closure: dict[str, set[str]] = {}
    for start in direct:
        seen: set[str] = set()
        stack = [start]
        while stack:
            for following in direct.get(stack.pop(), ()):
                if following not in seen:
                    seen.add(following)
                    stack.append(following)
        closure[start] = seen
    return closure


def _ordered(closure: dict[str, set[str]], a: str, b: str) -> bool:
    return b in closure.get(a, set()) or a in closure.get(b, set())


def check_merge_order(
    tickets: set[str],
    landed: set[str],
    edges: list[tuple[str, str]],
    overlap: list[dict[str, object]],
) -> list[str]:
    """Return a merge order; fail on a cycle or an unordered open overlapping pair."""
    for first, second in edges:
        if first not in tickets or second not in tickets:
            raise SystemExit(f"serialization edge names an unknown ticket: {first} -> {second}")
        if second in landed and first not in landed:
            raise SystemExit(f"serialization edge puts open {first} before landed {second}")
    remaining = {ticket: {first for first, second in edges if second == ticket} for ticket in tickets}
    order: list[str] = []
    while remaining:
        ready = sorted((t for t, preds in remaining.items() if not preds), key=lambda t: (t not in landed, t))
        if not ready:
            raise SystemExit(f"serialization graph has a cycle among {sorted(remaining)}")
        chosen = ready[0]
        order.append(chosen)
        del remaining[chosen]
        for preds in remaining.values():
            preds.discard(chosen)
    closure = _reachable(edges)
    unordered = [
        f"{pair['a']} / {pair['b']}"
        for pair in overlap
        if pair["a"] not in landed and pair["b"] not in landed and not _ordered(closure, str(pair["a"]), str(pair["b"]))
    ]
    if unordered:
        raise SystemExit(f"overlapping open pairs without ordering path: {'; '.join(unordered)}")
    return order


def check_lanes(
    lanes: list[dict[str, object]],
    write_sets: dict[str, list[str]],
    edges: list[tuple[str, str]],
    open_tickets: set[str],
    w3_tickets: set[str],
) -> None:
    """A lane is a group that may run in parallel: capped, disjoint, mutually independent."""
    closure = _reachable(edges)
    problems: list[str] = []
    names = [str(lane["lane"]) for lane in lanes]
    problems.extend(f"lane {name}: duplicate lane name" for name in sorted(set(names)) if names.count(name) > 1)
    placed: list[str] = []
    for lane in lanes:
        name, members = lane["lane"], list(lane["tickets"])
        if len(members) > LANE_CAP:
            problems.append(f"lane {name}: {len(members)} tickets exceed the cap of {LANE_CAP}")
        for position, a in enumerate(members):
            if a not in open_tickets:
                problems.append(f"lane {name}: {a} is not an open train ticket")
            if a in placed:
                problems.append(f"lane {name}: {a} is placed twice")
            placed.append(a)
            for b in members[position + 1 :]:
                if any(paths_overlap(x, y) for x in write_sets.get(a, []) for y in write_sets.get(b, [])):
                    problems.append(f"lane {name}: {a} and {b} share write paths")
                if _ordered(closure, a, b):
                    problems.append(f"lane {name}: {a} and {b} depend on each other")
    problems.extend(f"lane: open ticket {t} is in no lane" for t in sorted(open_tickets - set(placed)))
    if PT_WA_TICKET in open_tickets:
        problems.extend(
            f"lane: W3 ticket {t} has no ordering path from {PT_WA_TICKET} (PT-WA)"
            for t in sorted(w3_tickets - {PT_WA_TICKET})
            if t not in closure.get(PT_WA_TICKET, set())
        )
    if problems:
        raise SystemExit("; ".join(problems))


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")


def ticket_regression(
    ticket: str, spec: dict, inventory: list[dict[str, object]], write_set: list[str], tree: frozenset[str]
) -> dict[str, object]:
    """Regression commands of one ticket: owned rows, write-set test files and bound tests, all present at the pin."""
    bound = list(spec.get("regression_tests", []))
    absent = [t for t in bound if t.partition("::")[0] not in tree]
    if absent:
        raise SystemExit(f"{ticket}: regression tests absent at the read pin: {absent}")
    selected = set(bound)
    for row in inventory:
        if row["owner_ticket"] == ticket:
            selected.update(row.get("regression_tests", []))
    for pattern in write_set:
        kind, value = _pattern_kind(pattern)
        if kind == "file":
            matches = [pattern] if pattern in tree else []
        elif kind == "dir":
            matches = [f for f in tree if f.startswith(value)]
        else:
            matches = [f for f in tree if fnmatch.fnmatchcase(f, value)]
        selected.update(f for f in matches if _is_test_file(f))
    tests = sorted(t for t in selected if "::" not in t or t.partition("::")[0] not in selected)
    commands = [f"uv run pytest {' '.join(tests)} -q"] if tests else []
    commands.extend(spec.get("regression_extra_commands", []))
    reason = spec.get("regression_absent_reason")
    if not commands and not reason:
        raise SystemExit(f"{ticket}: no regression command and no regression_absent_reason")
    return {
        "regression_tests": tests,
        "regression_commands": commands,
        "regression_source": spec.get("regression_source"),
        "regression_absent_reason": None if commands else reason,
    }


def _ticket_section(bindings: dict, tree: frozenset[str], inventory: list[dict[str, object]]) -> dict[str, object]:
    specs: dict[str, dict] = bindings.get("tickets", {})
    if not specs:
        return {"tickets": [], "overlap_matrix": [], "serialization_edges": [], "lane_order": [], "merge_order": []}
    unknown = sorted({str(r["owner_ticket"]) for r in inventory if r["owner_ticket"]} - set(specs))
    if unknown:
        raise SystemExit(f"inventory owner tickets missing from bindings: {unknown}")

    entries: dict[str, dict[str, object]] = {}
    write_sets: dict[str, list[str]] = {}
    for ticket, spec in sorted(specs.items()):
        plan = spec.get("plan")
        plan_path = _REPO / plan if plan else None
        if plan_path is not None and not plan_path.is_file():
            raise SystemExit(f"{ticket}: plan file missing: {plan}")
        write_spec = spec["write_set"]
        if plan_path is None and write_spec.get("sections"):
            raise SystemExit(f"{ticket}: write-set sections require a plan file")
        text = plan_path.read_text(encoding="utf-8") if plan_path else ""
        try:
            extracted = extract_write_set(text, write_spec)
        except SystemExit as exc:
            raise SystemExit(f"{ticket}: {exc}") from exc
        write_sets[ticket] = extracted["paths"]
        entries[ticket] = {
            "ticket": ticket,
            "lifecycle": spec["lifecycle"],
            "landed": bool(spec.get("landed", False)),
            "wave": spec.get("wave", "W3"),
            "plan": plan,
            "plan_state": "PLAN" if plan else "NO_PLAN",
            "write_set_quality": spec["quality"],
            "write_set": [{"path": p, "state": label_path(p, tree)} for p in extracted["paths"]],
            "prose_mentions_not_in_write_set": extracted["prose_mentions"],
            "write_set_adjustments": {
                key: write_spec[key] for key in ("rewrites", "exclude_paths", "extra_paths") if write_spec.get(key)
            },
            "mandatory_tests": spec.get("mandatory_tests", []),
            "note": spec.get("note"),
            "code_fix_rows": [r["frz_id"] for r in inventory if r["owner_ticket"] == ticket],
            **ticket_regression(ticket, spec, inventory, extracted["paths"], tree),
        }

    edges: dict[tuple[str, str], str] = {}
    for ticket, spec in sorted(specs.items()):
        for predecessor, reason in spec.get("hard_predecessors", []):
            edges[(predecessor, ticket)] = reason
    for first, second, reason in bindings.get("serialization_edges", []):
        edges.setdefault((first, second), reason)

    landed = {t for t, e in entries.items() if e["landed"]}
    open_tickets = set(entries) - landed
    overlap = overlap_matrix(write_sets)
    order = check_merge_order(set(entries), landed, list(edges), overlap)
    lanes = bindings.get("lanes", [])
    check_lanes(lanes, write_sets, list(edges), open_tickets, {t for t in open_tickets if entries[t]["wave"] == "W3"})
    lane_of = {t: lane["lane"] for lane in lanes for t in lane["tickets"]}
    for ticket, entry in entries.items():
        entry["lane"] = lane_of.get(ticket)
        entry["merge_position"] = order.index(ticket) + 1
        entry["hard_predecessors"] = sorted(first for first, second in edges if second == ticket)
    return {
        "tickets": [entries[t] for t in order],
        "overlap_matrix": overlap,
        "serialization_edges": [{"first": a, "second": b, "reason": edges[(a, b)]} for a, b in sorted(edges)],
        "lane_order": [[lane["lane"], lane["tickets"], lane["label"]] for lane in lanes],
        "merge_order": order,
    }


# --- red-first ------------------------------------------------------------------

_OPERATOR_DISPOSITIONS = frozenset({"RED_FIRST_OUT", "RED_FIRST_LATER_WAVE"})
_SHA_BOUND_STATUSES = frozenset({"RED_AT_PREFIX", "RED_FIRST_UNREPRODUCIBLE"})
_RECORD_STATUSES = frozenset({"RED_AT_PREFIX", "EVIDENCE_MISSING"})
_CORRECTABLE = {"red_first_evidence_ref": "red_first_superseded_ref", "red_first_reason": "red_first_superseded_reason"}


def _empty_red_first() -> dict[str, object]:
    return {"red_first_prefix_sha": None, "red_first_status": "EVIDENCE_MISSING", "red_first_evidence_ref": None}


def _load_prior_red_first() -> dict[str, dict[str, object]]:
    """Every red-first field of every prior row; dispositions are never dropped."""
    if not _RW_PLAN_JSON.is_file():
        return {}
    try:
        data = json.loads(_RW_PLAN_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"prior RW plan is not valid JSON; refusing to drop red-first state: {exc}") from exc
    out: dict[str, dict[str, object]] = {}
    for row in data.get("code_fix_inventory") or []:
        if not isinstance(row, dict) or not row.get("frz_id") or not row.get("red_first_status"):
            continue
        prior = {k: v for k, v in row.items() if k.startswith("red_first_") or k.startswith("operator_")}
        status, sha = prior["red_first_status"], prior.get("red_first_prefix_sha")
        if status in _SHA_BOUND_STATUSES and not (
            isinstance(sha, str) and len(sha) == 40 and prior.get("red_first_evidence_ref")
        ):
            continue
        if status == "EVIDENCE_MISSING":
            prior["red_first_prefix_sha"] = None
        out[str(row["frz_id"])] = prior
    return out


def _load_record(bindings: dict) -> tuple[str | None, dict[str, dict[str, object]]]:
    name = bindings.get("red_first_record")
    if not name:
        return None, {}
    path = Path(name) if Path(name).is_absolute() else _REPO / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return _rel(path), {str(row["frz_id"]): row for row in data["rows"]}


def _apply_record(frz: str, current: dict[str, object], record: dict[str, object], ref: str) -> dict[str, object]:
    status = record["status"]
    if current["red_first_status"] in _OPERATOR_DISPOSITIONS:
        if status != current["red_first_status"]:
            raise SystemExit(f"{frz}: the red-first record cannot change an Operator disposition")
        return current
    if status not in _RECORD_STATUSES:
        raise SystemExit(f"{frz}: red-first record status {status!r} is not allowed")
    sha = record.get("red_first_prefix_sha") if status == "RED_AT_PREFIX" else None
    if status == "RED_AT_PREFIX" and not (isinstance(sha, str) and len(sha) == 40):
        raise SystemExit(f"{frz}: RED_AT_PREFIX record row needs a full prefix sha")
    if not record.get("reason"):
        raise SystemExit(f"{frz}: red-first record row needs a reason")
    new_ref = f"{ref}#{frz}"
    out: dict[str, object] = {
        "red_first_prefix_sha": sha,
        "red_first_status": status,
        "red_first_evidence_ref": new_ref,
        "red_first_reason": record["reason"],
    }
    old_ref = current.get("red_first_evidence_ref")
    superseded = old_ref if old_ref and old_ref != new_ref else current.get("red_first_superseded_ref")
    if superseded:
        out["red_first_superseded_ref"] = superseded
    if record.get("due_ticket"):
        out["red_first_due_ticket"] = record["due_ticket"]
    return out


def _apply_correction(current: dict[str, object], correction: dict[str, object]) -> dict[str, object]:
    """Correct an evidence reference or reason text; the status never changes."""
    out = dict(current)
    for key, superseded_key in _CORRECTABLE.items():
        if key in correction and out.get(key) != correction[key]:
            if out.get(key):
                out[superseded_key] = out[key]
            out[key] = correction[key]
    unknown = set(correction) - set(_CORRECTABLE) - {"why"}
    if unknown:
        raise SystemExit(f"record correction may not set {sorted(unknown)}")
    out["red_first_correction"] = correction["why"]
    return out


# --- inventory ------------------------------------------------------------------


def _negative_tests(frz: str, row: CodeFixRow, binding: dict, pin: str, tree: frozenset[str]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for entry in binding.get("negative_tests", []):
        missing = [k for k in NEGATIVE_TEST_ENTRY_KEYS if not entry.get(k)]
        if missing:
            raise SystemExit(f"{frz}: negative test entry misses {missing}")
        file_path, _, function = str(entry["node"]).partition("::")
        out = dict(entry)
        out["file_state_at_pin"] = "exists" if file_path in tree else "absent"
        if function and file_path in tree:
            source = _pin_text(pin, file_path)
            name = function.split("[", 1)[0].split("::")[-1]
            if source is None:
                out["function_state_at_pin"] = "unknown"
            else:
                out["function_state_at_pin"] = "exists" if f"def {name}(" in source else "absent"
        elif function:
            out["function_state_at_pin"] = "absent"
        entries.append(out)
    result: dict[str, object] = {"negative_test_id": row.negative_test, "negative_tests": entries}
    if row.negative_test is None or row.negative_test in CLASS_MARKERS:
        if not entries:
            raise SystemExit(f"{frz}: matrix gives {row.negative_test!r}; bindings must name the negative test")
        result["negative_test_id"] = entries[0]["nt_id"]
        result["matrix_negative_test_marker"] = row.negative_test
    return result


def _regression(
    frz: str, row: CodeFixRow, binding: dict, appendix_c: dict[str, str], tree: frozenset[str]
) -> dict[str, object]:
    sources = ["matrix_appendix_a_tests" if row.regression_scope else "MAPPING_MISSING"]
    out: dict[str, object] = {"regression_scope": row.regression_scope or "MAPPING_MISSING"}
    if appendix_c.get("regression scope"):
        out["regression_scope_appendix_c"] = appendix_c["regression scope"]
        sources.append("matrix_appendix_c")
    tests = list(binding.get("regression_tests", []))
    absent = [t for t in tests if label_path(t.partition("::")[0], tree) == "new"]
    if absent:
        raise SystemExit(f"{frz}: regression tests absent at the read pin: {absent}")
    if tests:
        sources.append("rw_plan_bindings")
    out["regression_tests"] = tests
    out["regression_command"] = f"uv run pytest {' '.join(tests)} -q" if tests else None
    out["regression_scope_source"] = sources
    return out


def _names_row(path: Path, frz: str) -> bool:
    short = frz.removeprefix("FRZ-")
    return re.search(rf"\b(?:FRZ-)?{re.escape(short)}\b", path.read_text(encoding="utf-8")) is not None


def _traceability(ticket: str, frz: str) -> dict[str, object]:
    ticket_file = _TICKETS / f"{ticket}.md"
    plan_file = _PLANS / f"PLAN_{ticket}.md"
    return {
        "ticket_file": _rel(ticket_file) if ticket_file.is_file() else None,
        "ticket_file_names_row": ticket_file.is_file() and _names_row(ticket_file, frz),
        "plan_file": _rel(plan_file) if plan_file.is_file() else None,
        "plan_names_row": plan_file.is_file() and _names_row(plan_file, frz),
    }


def _inventory(
    bindings: dict, tree: frozenset[str], statuses: dict[str, str]
) -> tuple[list[dict[str, object]], str | None]:
    pin = bindings["read_pin"]
    rows = sorted(parse_appendix_a_code_fix_inventory(), key=lambda r: r.frz_id)
    ids = {r.frz_id for r in rows}
    row_bindings: dict[str, dict] = bindings.get("code_fix_rows", {})
    corrections: dict[str, dict] = bindings.get("record_corrections", {})
    record_ref, records = _load_record(bindings)
    for label, keys in (("code_fix_rows", row_bindings), ("record_corrections", corrections), ("record", records)):
        if set(keys) - ids:
            raise SystemExit(f"{label} names rows outside the inventory: {sorted(set(keys) - ids)}")
    prior = _load_prior_red_first()
    appendix_c = parse_appendix_c()

    inventory: list[dict[str, object]] = []
    for row in rows:
        frz = row.frz_id
        binding = row_bindings.get(frz, {})
        red_first = prior.get(frz) or _empty_red_first()
        if frz in records:
            red_first = _apply_record(frz, red_first, records[frz], str(record_ref))
        if frz in corrections:
            red_first = _apply_correction(red_first, corrections[frz])
        ownership = apply_explicit_ownership(frz, row)
        negative = _negative_tests(frz, row, binding, pin, tree)
        entry: dict[str, object] = {
            "frz_id": frz,
            "summary": row.summary,
            "matrix_status": statuses.get(frz),
            "owner_ticket": ownership.get("owner_ticket"),
            **negative,
            **_empty_red_first(),
            **red_first,
            **_regression(frz, row, binding, appendix_c.get(frz, {}), tree),
            "delta_findings": row.delta_findings,
            "remediation_track": row.remediation_track,
        }
        entry.update({k: v for k, v in ownership.items() if k != "owner_ticket"})
        if entry["owner_ticket"]:
            entry["traceability"] = _traceability(str(entry["owner_ticket"]), frz)
        if binding.get("note"):
            entry["note"] = binding["note"]
        inventory.append(entry)
    return inventory, record_ref


def _reconciliation(inventory: list[dict[str, object]], appendix_c_ids: set[str]) -> dict[str, object]:
    ids = {str(r["frz_id"]) for r in inventory}
    only_c = sorted(appendix_c_ids - ids)
    only_a = sorted(ids - appendix_c_ids)
    named = sum(1 for r in inventory if r["negative_tests"] and r["negative_test_id"] not in (None, *CLASS_MARKERS))
    return {
        "mep_historical_unplanned_count": 35,
        "matrix_appendix_a_code_fix_count": len(ids),
        "matrix_appendix_c_block_count": len(appendix_c_ids),
        "appendix_c_not_code_fix": only_c,
        "code_fix_without_appendix_c_block": only_a,
        "arithmetic": f"{len(ids)} = {len(appendix_c_ids)} - {len(only_c)} + {len(only_a)}",
        "delta_explanation": (
            "The CODE_FIX inventory is every Appendix A row whose required_fix starts with CODE_FIX. "
            "Appendix C is a block list with a different membership; both differences are listed."
        ),
        "find_af2_23_note": (
            "FIND-AF2-23 (18 CODE_FIX without explicit negative tests) is the historical WP-12 phase check 3; "
            f"rows with a named negative test in this plan: {named} / {len(ids)}. "
            "A name is not red-first evidence; see the red-first column."
        ),
    }


def _audit_ids(bindings: dict, tree: frozenset[str]) -> dict[str, object]:
    source = bindings.get("audit_ids_18")
    if not source:
        return {}
    rows: list[dict[str, object]] = []
    for row in source["rows"]:
        paths = [{"path": p, "state": label_path(p, tree)} for p in row["test_paths"]]
        present = [p["path"] for p in paths if p["state"] == "exists"]
        entry = {k: v for k, v in row.items() if k != "test_paths"}
        entry["tests"] = paths
        entry["regression_command"] = f"uv run pytest {' '.join(present)} -q" if present else None
        rows.append(entry)
    return {"source": source["source"], "source_sha256": source["source_sha256"], "rows": rows}


def build() -> dict[str, object]:
    bindings = load_bindings()
    tree = _pin_tree(bindings["read_pin"])
    all_rows = parse_appendix_a_all_rows()
    statuses = {str(r["frz_id"]): str(r["matrix_status"]) for r in all_rows}
    mapping_errors = find_af2_errors(bindings.get("find_af2"))
    if mapping_errors:
        raise SystemExit("; ".join(mapping_errors))
    inventory, record_ref = _inventory(bindings, tree, statuses)
    ids = {str(r["frz_id"]) for r in inventory}
    payload: dict[str, object] = {
        "schema": "rw_plan.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_sha": _git_head(),
        "read_pin": bindings["read_pin"],
        "bindings_sha256": hashlib.sha256(_BINDINGS_JSON.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "node": "T-AF2-RW-PLAN",
        "notes": bindings.get("md_notes", []),
        "reconciliation": _reconciliation(inventory, set(parse_appendix_c())),
        "status_semantics": bindings.get("status_semantics", []),
        "red_first_record": record_ref,
        "code_fix_inventory": inventory,
        "coverage": build_coverage(all_rows, ids, bindings.get("coverage_overrides", {})),
        "find_af2": bindings["find_af2"],
        "audit_flag_conflicts": AUDIT_FLAG_CONFLICTS,
        "audit_ids_18": _audit_ids(bindings, tree),
        "lane_cap": LANE_CAP,
        "active_writer_limit": ACTIVE_WRITER_LIMIT,
        **_ticket_section(bindings, tree, inventory),
        "path_b_invariants": bindings["path_b_invariants"],
        "g2_entry_conditions": bindings["g2_entry_conditions"],
        "held_paths": bindings.get("held_paths", []),
        "pre_g2_order": bindings.get("pre_g2_order", []),
        "dependencies": {**bindings["dependencies"], "SCOPE": list(SCOPE_HARD_PREDECESSORS)},
        "implementation_disposition": bindings["implementation_disposition"],
        "versioned_plan_record": bindings.get("versioned_plan_record"),
    }
    return payload


# --- markdown ---------------------------------------------------------------------


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return lines


def owner_display(row: dict) -> str:
    """The one owner rendering; it reads the same fields the JSON carries."""
    parts: list[str] = []
    if row.get("owner_ticket"):
        parts.append(str(row["owner_ticket"]))
    if row.get("owner_node"):
        parts.append(f"node {row['owner_node']} ({row.get('owner_wave')})")
    if row.get("verification_node"):
        parts.append(f"verify {row['verification_node']} ({row.get('verification_wave')})")
    if row.get("owner_ticket_family"):
        parts.append(f"ticket family {row['owner_ticket_family']}")
    return "; ".join(parts) or "UNASSIGNED"


def _bullets(values: object) -> list[str]:
    return [f"- {value}" for value in values or []] or ["- none"]


def _render_md(payload: dict) -> str:
    inventory = payload["code_fix_inventory"]
    recon = payload["reconciliation"]
    lines = [
        "# RW-PLAN — T-AF2-RW-PLAN (canonical)",
        "",
        "Generated by `scripts/governance/build_rw_plan.py`. Do not edit by hand.",
        "",
        f"**Generated:** {payload['generated_at']}",
        f"**Execution SHA:** `{payload['execution_sha']}`",
        f"**Read pin (file states and measurements):** `{payload['read_pin']}`",
        f"**Bindings sha256:** `{payload['bindings_sha256']}`",
        "**Node:** T-AF2-RW-PLAN (W2, DOC)",
        "",
    ]
    for note in payload["notes"]:
        lines.extend([note, ""])
    lines.extend(
        [
            "## 0. Reconciliation",
            "",
            f"- MEP historical unplanned pack: **{recon['mep_historical_unplanned_count']}**",
            f"- Arithmetic: **{recon['arithmetic']}** (CODE_FIX rows = Appendix C blocks - only in C + only in A)",
            f"- Appendix C blocks that are not CODE_FIX rows: {', '.join(recon['appendix_c_not_code_fix']) or 'none'}",
            "- CODE_FIX rows without an Appendix C block: "
            f"{', '.join(recon['code_fix_without_appendix_c_block']) or 'none'}",
            f"- {recon['delta_explanation']}",
            f"- {recon['find_af2_23_note']}",
        ]
    )
    lines.extend(f"- {s}" for s in payload["status_semantics"])
    lines.extend(["", "## 1. CODE_FIX inventory", ""])
    lines.extend(
        _table(
            ["FRZ", "Matrix status", "Owner", "Negative test", "Red-first", "Prefix SHA", "Findings"],
            [
                [
                    f"`{r['frz_id']}`",
                    r["matrix_status"],
                    owner_display(r),
                    r["negative_test_id"],
                    r["red_first_status"],
                    r.get("red_first_prefix_sha") or "—",
                    ", ".join(r["delta_findings"]) or "—",
                ]
                for r in inventory
            ],
        )
    )
    lines.extend(["", "## 2. Negative tests (node, mechanism, owner; file state at the read pin)", ""])
    lines.extend(
        _table(
            ["FRZ", "NT id", "Node", "File", "Function", "Owner ticket", "Mechanism"],
            [
                [
                    f"`{r['frz_id']}`",
                    t["nt_id"],
                    f"`{t['node']}`",
                    t["file_state_at_pin"],
                    t.get("function_state_at_pin", "—"),
                    t["owner_ticket"],
                    t["mechanism"],
                ]
                for r in inventory
                for t in r["negative_tests"]
            ],
        )
    )
    lines.extend(["", "## 3. Red-first evidence", ""])
    lines.extend(
        _table(
            ["FRZ", "Status", "Evidence", "Due ticket", "Reason"],
            [
                [
                    f"`{r['frz_id']}`",
                    r["red_first_status"],
                    r.get("red_first_evidence_ref") or "—",
                    r.get("red_first_due_ticket") or "—",
                    r.get("red_first_reason") or "—",
                ]
                for r in inventory
            ],
        )
    )
    lines.extend(["", "## 4. Regression scope and commands", ""])
    for r in inventory:
        lines.append(f"### `{r['frz_id']}`")
        lines.append(f"- Matrix tests column: {r['regression_scope']}")
        if r.get("regression_scope_appendix_c"):
            lines.append(f"- Matrix Appendix C: {r['regression_scope_appendix_c']}")
        lines.append(f"- Sources: {', '.join(r['regression_scope_source'])}")
        lines.append(f"- Command: `{r['regression_command']}`" if r["regression_command"] else "- Command: none bound")
        if r.get("note"):
            lines.append(f"- Note: {r['note']}")
        lines.append("")
    lines.extend(["## 5. Audit-flag conflicts", ""])
    lines.extend(
        _table(
            ["Finding", "Owner", "Disposition"],
            [[c["finding_id"], owner_display(c), c["disposition"]] for c in payload["audit_flag_conflicts"]],
        )
    )
    audit_ids = payload["audit_ids_18"]
    if audit_ids:
        lines.extend(
            [
                "",
                "## 6. CODE_FIX finding ids of WP-12 phase check 3",
                "",
                f"Source (untracked): `{audit_ids['source']}`, sha256 `{audit_ids['source_sha256']}`.",
                "",
            ]
        )
        lines.extend(
            _table(
                ["Id", "Finding", "Node", "Owner ticket", "Negative test", "Test files (state at pin)", "Command"],
                [
                    [
                        r["id"],
                        r["finding"],
                        r["node"],
                        r["owner_ticket"],
                        r["negative_test"],
                        ", ".join(f"`{t['path']}` ({t['state']})" for t in r["tests"]),
                        f"`{r['regression_command']}`" if r["regression_command"] else "authored by the owner ticket",
                    ]
                    for r in audit_ids["rows"]
                ],
            )
        )
    lines.extend(
        [
            "",
            "## 7. Train tickets",
            "",
            f"A lane is a group that may run in parallel (cap {payload['lane_cap']}, disjoint write sets, no "
            f"dependency between members). At most {payload['active_writer_limit']} writers are active at once (D-4).",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Merge", "Ticket", "Lane", "Lifecycle", "Landed", "Hard predecessors", "Rows", "Write set", "Tests"],
            [
                [
                    t["merge_position"],
                    t["ticket"],
                    t["lane"] or "—",
                    t["lifecycle"],
                    "yes" if t["landed"] else "no",
                    ", ".join(t["hard_predecessors"]) or "—",
                    ", ".join(t["code_fix_rows"]) or "—",
                    f"{len(t['write_set'])} paths, {t['write_set_quality']}, {t['plan_state']}",
                    "; ".join(t["mandatory_tests"]) or "—",
                ]
                for t in payload["tickets"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Per-ticket regression commands",
            "",
            "Each command is the union of the regression tests of the rows the ticket owns, the test files of its "
            "write set that exist at the read pin and the tests bound to the ticket. Planned new test files are not "
            "listed here (see section 2 and the Tests column). Paths are those of the read pin; a ticket that "
            "renames or deletes a listed file runs the replacement.",
            "",
        ]
    )
    for t in payload["tickets"]:
        lines.append(f"#### {t['ticket']}")
        if t["regression_source"]:
            lines.append(f"Bound tests: {t['regression_source']}")
        if t["regression_absent_reason"]:
            lines.append(f"No command: {t['regression_absent_reason']}")
        lines.extend(f"- `{command}`" for command in t["regression_commands"])
        lines.append("")
    lines.extend(["## 8. Lanes", ""])
    lines.extend(
        _table(["Lane", "Label", "Tickets"], [[e[0], e[2], ", ".join(e[1]) or "—"] for e in payload["lane_order"]])
    )
    lines.extend(["", "## 9. Serialization edges (first merges before second)", ""])
    lines.extend(
        _table(
            ["First", "Second", "Reason"],
            [[e["first"], e["second"], e["reason"]] for e in payload["serialization_edges"]],
        )
    )
    position = {t: i for i, t in enumerate(payload["merge_order"])}
    landed = {t["ticket"] for t in payload["tickets"] if t["landed"]}
    lines.extend(["", "## 10. Overlap matrix (every overlapping pair, every shared path)", ""])
    lines.extend(
        _table(
            ["Pair (merge order)", "Ordered by", "Shared paths"],
            [
                [
                    " → ".join(sorted((p["a"], p["b"]), key=lambda t: position[t])),
                    "history (landed)" if {p["a"], p["b"]} & landed else "serialization graph",
                    ", ".join(f"`{x}`" for x in p["paths"]),
                ]
                for p in payload["overlap_matrix"]
            ],
        )
    )
    lines.extend(["", "## 11. Write sets (complete)", ""])
    for t in sorted(payload["tickets"], key=lambda e: e["ticket"]):
        lines.append(f"### {t['ticket']}")
        lines.append("")
        source = f"`{t['plan']}`" if t["plan"] else "no plan file; the listed paths are indicative"
        lines.append(f"Source: {source}. Quality: {t['write_set_quality']}.")
        if t.get("note"):
            lines.append(str(t["note"]))
        lines.append("")
        lines.extend(f"- `{p['path']}` ({p['state']})" for p in t["write_set"])
        for kind, items in t["write_set_adjustments"].items():
            for item in items:
                what = item.get("path") or f"{item['from']} → {item['to']}"
                lines.append(f"- adjustment `{kind}`: `{what}` — {item['reason']}")
        if t["prose_mentions_not_in_write_set"]:
            mentions = ", ".join(f"`{p}`" for p in t["prose_mentions_not_in_write_set"])
            lines.append(f"- mentioned in section prose, not counted: {mentions}")
        lines.append("")
    lines.extend(
        ["## 12. OPEN matrix rows outside the CODE_FIX inventory", "", f"Rows: {len(payload['coverage'])}.", ""]
    )
    lines.extend(
        _table(
            ["FRZ", "Why outside the inventory (required_fix)", "Owning nodes", "Accountable", "Owner ticket", "Note"],
            [
                [
                    f"`{c['frz_id']}`",
                    c["reason"],
                    ", ".join(c["owner_nodes"]),
                    c["accountable"] or "not stated in matrix",
                    c.get("owner_ticket") or "—",
                    c.get("note") or "—",
                ]
                for c in payload["coverage"]
            ],
        )
    )
    lines.extend(["", "## 13. FIND-AF2 mapping", ""])
    lines.extend(
        _table(
            ["Finding", "In train", "Mapping"],
            [[e["id"], "yes" if e.get("in_train") else "no", e["mapping"]] for e in payload["find_af2"]],
        )
    )
    lines.extend(["", "## 14. Path B and tenant isolation", "", *_bullets(payload["path_b_invariants"])])
    lines.extend(["", "## 15. G2 entry conditions", "", *_bullets(payload["g2_entry_conditions"])])
    lines.extend(["", "## 16. Held paths from G2", "", *_bullets(payload["held_paths"])])
    lines.extend(["", "## 17. Order before G2", "", *_bullets(payload["pre_g2_order"])])
    lines.extend(["", "## 18. Dependencies", ""])
    lines.extend(f"- {key}: {', '.join(value)}" for key, value in payload["dependencies"].items())
    lines.extend(["", "## 19. Implementation disposition", ""])
    for key, value in payload["implementation_disposition"].items():
        lines.append(f"- {key}: {', '.join(value) if isinstance(value, list) else value}")
    lines.extend(
        [
            "",
            "## 20. Binding record of cited plan versions",
            "",
            str(payload["versioned_plan_record"] or "none"),
            "",
            "## 21. Machine contract",
            "",
            "Strict phase check 3: `scripts/governance/verify_rw_plan_phase_check3.py`",
            "",
            f"JSON SSOT: `rw_plan.v1.json` (schema `{payload['schema']}`); reviewed input: `rw_plan_bindings.v1.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build()
    _RW_PLAN_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    _RW_PLAN_MD.write_text(_render_md(payload), encoding="utf-8", newline="\n")
    print(f"[OK] Wrote {_RW_PLAN_MD.relative_to(_REPO)}")
    print(f"[OK] Wrote {_RW_PLAN_JSON.relative_to(_REPO)} ({len(payload['code_fix_inventory'])} CODE_FIX rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
