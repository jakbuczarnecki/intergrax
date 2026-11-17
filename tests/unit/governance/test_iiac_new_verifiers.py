"""Corpus and negative tests for the PT-E04 verifiers (ticket T-IGX-044-E).

Covers ``verify_iiac_manifest_artifacts.py``, ``verify_iiac_manifest_corpus.py`` and
``verify_iiac_delegation_refs.py``. The data lives in ``tests/fixtures/iiac_verifier_corpus/``:
``cases.json`` names, for every case, its polarity, the expected exit code, the expected finding codes
and (for negative cases) the weakness of the existing validators it targets.

The tests build a throw-away git repository, so they touch nothing in the repository under test, and
they never call a tracked-path writer: the verifiers under test are read-only.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.governance import (
    iiac_activation_gate,
    iiac_verifier_common,
    verify_iiac_delegation_refs,
    verify_iiac_manifest_artifacts,
    verify_iiac_manifest_corpus,
)

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts/governance"
CORPUS = REPO / "tests/fixtures/iiac_verifier_corpus"
NEW_SCRIPTS = (
    "iiac_verifier_common.py",
    "verify_iiac_manifest_corpus.py",
    "verify_iiac_manifest_artifacts.py",
    "verify_iiac_delegation_refs.py",
)
NEW_PATHS = (*(f"scripts/governance/{n}" for n in NEW_SCRIPTS), "scripts/governance/iiac_known_draft_manifests.v1.json")
TS = "2026-11-16T10:00:00+01:00"

_CASES = json.loads((CORPUS / "cases.json").read_text(encoding="utf-8"))["cases"]


# --------------------------------------------------------------------------- throw-away repository


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_DATE": "2026-11-16T09:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-11-16T09:00:00+00:00",
    }
    proc = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "user.name=corpus",
            "-c",
            "user.email=corpus@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **env},
    )
    return proc.stdout.strip()


def _commit(repo: Path, files: dict[str, bytes], message: str) -> str:
    for rel, data in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        _git(repo, "add", "--", rel)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _delegation_ticket(index: int, case: dict[str, Any]) -> tuple[str, str]:
    ticket_id = f"T-IGX-900-D{index:02d}"
    lines = [
        f"# {ticket_id} - corpus ticket",
        "",
        case.get("extra_lines", ""),
        "",
        "## Approved",
        "",
        case["approval_line"],
        *case.get("after_lines", []),
        "",
    ]
    return ticket_id, "\n".join(lines)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    repo = tmp_path_factory.mktemp("iiac_corpus_repo")
    _git(repo, "init", "-q", "-b", "main")
    fixtures = {p.relative_to(CORPUS).as_posix(): p.read_bytes() for p in (CORPUS / "evidence").iterdir()}
    delegations = {p.name: p.read_bytes() for p in (CORPUS / "delegations").iterdir()}
    shas: dict[str, str] = {}
    shas["C_IMPL"] = _commit(repo, {"src/impl.txt": b"impl\n"}, "impl")
    _git(repo, "checkout", "-q", "-b", "side")
    shas["C_SIDE"] = _commit(repo, {"delegations/DELEG-SIDE.md": delegations["DELEG-SIDE.md"]}, "side delegation")
    _git(repo, "checkout", "-q", "main")
    shas["C_TEST"] = _commit(repo, {"tests/t.txt": b"t\n"}, "tests")
    shas["C_AUDIT"] = _commit(repo, {"audit/a.txt": b"a\n"}, "audit")
    shas["C_DELEG"] = _commit(
        repo,
        {
            "delegations/DELEG-GOOD.md": delegations["DELEG-GOOD.md"],
            "delegations/DELEG-OTHER.md": delegations["DELEG-OTHER.md"],
            "delegations/DELEG-SUBSTR.md": delegations["DELEG-SUBSTR.md"],
        },
        "delegations",
    )
    shas["C_EVID"] = _commit(repo, {rel: data for rel, data in fixtures.items()}, "evidence")
    shas["C_CERT"] = _commit(repo, {"cert/c.txt": b"c\n"}, "certification")
    tokens = {
        **shas,
        "ZERO": "0" * 40,
        "MISSING": "1234567890abcdef1234567890abcdef12345678",
        "D_REPORT": hashlib.sha256(fixtures["evidence/report.md"]).hexdigest(),
        "D_SCOPE": hashlib.sha256(fixtures["evidence/scope.json"]).hexdigest(),
        "D_MATRIX": hashlib.sha256(fixtures["evidence/matrix.txt"]).hexdigest(),
        "D_FINDINGS": hashlib.sha256(fixtures["evidence/findings.json"]).hexdigest(),
    }
    tickets: dict[str, Path] = {}
    ticket_files: dict[str, bytes] = {}
    delegation_cases = [c for c in _CASES if c["verifier"] == "delegation"]
    for index, case in enumerate(delegation_cases, start=1):
        case = {
            **case,
            "approval_line": _subst(case["approval_line"], tokens),
            "after_lines": _subst(case.get("after_lines", []), tokens),
        }
        ticket_id, text = _delegation_ticket(index, case)
        ticket_files[f"governance/tickets/{ticket_id}.md"] = text.encode()
        tickets[case["id"]] = repo / f"governance/tickets/{ticket_id}.md"
    _commit(repo, ticket_files, "tickets")
    return {"repo": repo, "tokens": tokens, "tickets": tickets}


def _subst(value: Any, tokens: dict[str, str]) -> Any:
    if isinstance(value, str):
        return re.sub(r"@@([A-Z_]+)@@", lambda m: tokens[m.group(1)], value)
    if isinstance(value, list):
        return [_subst(v, tokens) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v, tokens) for k, v in value.items()}
    return value


def _manifest(spec: dict[str, Any], tokens: dict[str, str]) -> dict[str, Any]:
    base = json.loads(_subst((CORPUS / "base_manifest.json.tmpl").read_text(encoding="utf-8"), tokens))
    base.update(_subst(spec.get("set", {}), tokens))
    for key in spec.get("delete", []):
        base.pop(key, None)
    return base


def _run(main: Any, argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(argv)
    return rc, err.getvalue() + out.getvalue()


def _run_case(case: dict[str, Any], world: dict[str, Any], workdir: Path) -> tuple[int, str]:
    repo: Path = world["repo"]
    tokens: dict[str, str] = world["tokens"]
    kind = case["verifier"]
    if kind == "artifacts":
        path = workdir / f"{case['id']}.json"
        path.write_text(json.dumps(_manifest(case, tokens)), encoding="utf-8", newline="\n")
        return _run(verify_iiac_manifest_artifacts.main, [str(path), "--repo", str(repo), *case["args"]])
    if kind == "delegation":
        ticket: Path = world["tickets"][case["id"]]
        return _run(verify_iiac_delegation_refs.main, [str(ticket), "--repo", str(repo), *case["args"]])
    assert kind == "corpus"
    base = repo / "cases" / case["id"]
    qual = base / "qual"
    qual.mkdir(parents=True)
    digests: dict[str, str] = {}
    for spec in case["tree"]:
        rel = f"cases/{case['id']}/qual/iiac_audit_session_manifest.{spec['name']}"
        data = (
            spec["raw"].encode() if "raw" in spec else (json.dumps(_manifest(spec, tokens), indent=2) + "\n").encode()
        )
        (repo / rel).write_bytes(data)
        digests[spec["name"]] = hashlib.sha256(data).hexdigest()
    entries = []
    for entry in case["allowlist"]:
        rel = f"cases/{case['id']}/qual/iiac_audit_session_manifest.{entry['name']}"
        digest = digests.get(entry["name"], "0" * 64) if entry["digest"] == "auto" else entry["digest"]
        entries.append({"path": rel, "sha256": digest, "reason": entry["reason"]})
    allow = case["allowlist_raw"] or {"schema": "iiac_known_draft_manifests.v1", "entries": entries}
    (base / "allow.json").write_text(json.dumps(allow), encoding="utf-8", newline="\n")
    if case["mutate_after_listing"]:
        spec = case["mutate_after_listing"]
        target = repo / f"cases/{case['id']}/qual/iiac_audit_session_manifest.{spec['name']}"
        changed = json.loads(target.read_text(encoding="utf-8"))
        changed.update(spec["set"])
        target.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8", newline="\n")
    argv = ["--repo", str(repo), "--root", f"cases/{case['id']}/qual", "--allowlist", f"cases/{case['id']}/allow.json"]
    for name in case["certifying"]:
        argv += ["--certifying", f"cases/{case['id']}/qual/iiac_audit_session_manifest.{name}"]
    return _run(verify_iiac_manifest_corpus.main, argv)


# --------------------------------------------------------------------------- the corpus


def test_corpus_is_well_formed() -> None:
    ids = [c["id"] for c in _CASES]
    assert len(ids) == len(set(ids))
    negatives = [c for c in _CASES if c["polarity"] == "negative"]
    positives = [c for c in _CASES if c["polarity"] == "positive"]
    assert len(negatives) >= 30 and len(positives) >= 5
    for case in negatives:
        assert len(case["weakness"].strip()) >= 30, f"{case['id']}: a negative case must name the weakness it targets"
        assert case["expect_exit"] in (1, 2)
    for case in positives:
        assert case["expect_exit"] == 0 and not case["weakness"]
    assert {c["verifier"] for c in _CASES} == {"artifacts", "delegation", "corpus"}


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_corpus_case(case: dict[str, Any], world: dict[str, Any], tmp_path: Path) -> None:
    rc, output = _run_case(case, world, tmp_path)
    assert rc == case["expect_exit"], output
    for code in case["expect_codes"]:
        assert re.search(rf"\b{code}\b", output), f"{code} not reported:\n{output}"
    for code in case["absent_codes"]:
        assert not re.search(rf"\b{code}\b", output), f"{code} must not be reported:\n{output}"
    if rc == 1:
        assert "FAIL" in output
    if rc == 0:
        assert "PASS" in output


# --------------------------------------------------------------------------- exit codes through the real entry point


def _script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / name), *args], cwd=REPO, capture_output=True, text=True, check=False
    )


def test_exit_codes_are_exact_when_run_as_scripts(world: dict[str, Any], tmp_path: Path) -> None:
    repo = world["repo"]
    tokens = world["tokens"]
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_manifest({}, tokens)), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_manifest({"set": {"verdict": "FAIL"}}, tokens)), encoding="utf-8")
    assert _script("verify_iiac_manifest_artifacts.py", str(good), "--repo", str(repo)).returncode == 0
    assert _script("verify_iiac_manifest_artifacts.py", str(bad), "--repo", str(repo)).returncode == 1
    assert _script("verify_iiac_manifest_artifacts.py", str(tmp_path / "absent.json")).returncode == 2
    assert _script("verify_iiac_manifest_artifacts.py", str(good), "--repo", str(tmp_path / "no_repo")).returncode == 2
    assert _script("verify_iiac_delegation_refs.py", str(tmp_path / "absent.md")).returncode == 2
    assert _script("verify_iiac_manifest_corpus.py", "--allowlist", str(tmp_path / "absent.json")).returncode == 2
    assert _script("verify_iiac_manifest_corpus.py", "--bogus-flag").returncode == 2


# --------------------------------------------------------------------------- the real repository


def _porcelain() -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def test_published_allowlist_is_exactly_the_three_known_drafts_and_corpus_passes() -> None:
    allow = json.loads((SCRIPTS / "iiac_known_draft_manifests.v1.json").read_text(encoding="utf-8"))
    assert allow["schema"] == "iiac_known_draft_manifests.v1"
    assert sorted(Path(e["path"]).name for e in allow["entries"]) == [
        "iiac_audit_session_manifest.EVOL-X-B4.reaudit_draft.json",
        "iiac_audit_session_manifest.EVOL-X-W4-01.reaudit_draft.json",
        "iiac_audit_session_manifest.EVOL-X.auditor_draft.json",
    ]
    before = _porcelain()
    rc, output = _run(verify_iiac_manifest_corpus.main, [])
    assert rc == 0, output
    match = re.search(r'"manifests": (\d+)', output)
    assert match and int(match.group(1)) >= 39
    assert _porcelain() == before, "the corpus verifier must not change the working tree"


def test_published_corpus_fails_without_the_allowlist(tmp_path: Path) -> None:
    empty = tmp_path / "empty_allowlist.json"
    empty.write_text(json.dumps({"schema": "iiac_known_draft_manifests.v1", "entries": []}), encoding="utf-8")
    rc, output = _run(verify_iiac_manifest_corpus.main, ["--allowlist", str(empty)])
    assert rc == 1
    assert len(re.findall(r"MANIFEST_INVALID", output)) == 3


@pytest.mark.parametrize(
    ("ticket", "expected_rc", "code"),
    [
        ("T-IGX-030-B.md", 0, None),
        ("T-IGX-030-A.md", 1, "DELEGATION_UNPINNED"),
        ("T-IGX-034-B.md", 1, "DELEGATION_UNRESOLVED"),
        ("T-IGX-035-E.md", 1, "DELEGATION_NOT_A_PATH"),
    ],
)
def test_replay_of_published_tickets_reproduces_the_known_delegation_defects(
    ticket: str, expected_rc: int, code: str | None
) -> None:
    path = REPO / "governance/tickets" / ticket
    if not path.is_file():
        pytest.skip(f"{ticket} not present in this checkout")
    rc, output = _run(verify_iiac_delegation_refs.main, [str(path)])
    assert rc == expected_rc, output
    if code:
        assert code in output


# --------------------------------------------------------------------------- scope guarantees


def test_new_files_are_not_material_and_not_imported_by_the_activation_gate() -> None:
    for rel in NEW_PATHS:
        assert rel not in iiac_activation_gate.MATERIAL_GOVERNANCE_PATHS
    module_names = [Path(n).stem for n in NEW_SCRIPTS]
    for material in sorted(iiac_activation_gate.MATERIAL_GOVERNANCE_PATHS):
        path = REPO / material
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        for name in module_names:
            assert name not in text, f"{material} references new module {name}"
    gate_text = (SCRIPTS / "iiac_activation_gate.py").read_text(encoding="utf-8")
    assert "iiac_known_draft_manifests" not in gate_text


def _call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


@pytest.mark.parametrize("name", NEW_SCRIPTS)
def test_new_scripts_are_stdlib_only_and_never_write(name: str) -> None:
    tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots = {node.module.split(".")[0]}
        else:
            continue
        for root in roots:
            assert root in sys.stdlib_module_names or root == "scripts", f"{name} imports non-stdlib {root}"
    forbidden = {
        "write_text",
        "write_bytes",
        "unlink",
        "rmdir",
        "mkdir",
        "rename",
        "touch",
        "rmtree",
        "copy",
        "copyfile",
        "move",
        "chmod",
    }
    assert not (_call_names(tree) & forbidden), f"{name} calls a filesystem writer"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
            assert not set(str(mode)) & set("wax+"), f"{name} opens a file for writing"


def test_git_wrapper_is_read_only() -> None:
    assert iiac_verifier_common._READ_ONLY_GIT <= {
        "cat-file",
        "merge-base",
        "rev-parse",
        "show",
        "log",
        "ls-tree",
        "ls-files",
    }
    for mutating in ("commit", "add", "checkout", "reset", "push", "fetch", "stash", "worktree", "config", "branch"):
        with pytest.raises(ValueError):
            iiac_verifier_common.run_git(REPO, mutating)


@pytest.mark.parametrize("ticket", ["T-IGX-044-A.md", "T-IGX-046-B.md", "T-IGX-042-A.md"])
def test_replay_of_published_two_line_tickets_resolves_the_documented_form(ticket: str) -> None:
    """The documented form (governance/tickets/README.md item 3): Approved line, Delegation line directly after."""
    path = REPO / "governance/tickets" / ticket
    if not path.is_file():
        pytest.skip(f"{ticket} not present in this checkout")
    text = path.read_text(encoding="utf-8")
    assert re.search(r"^Approved: operator-delegate @ \S+\nDelegation: \S+@[0-9a-f]{40}$", text, re.MULTILINE)
    rc, output = _run(verify_iiac_delegation_refs.main, [str(path)])
    assert rc == 0, output
    assert "DELEGATE_IDENTITY_UNKNOWN" in output  # no label in this form: reported, not hidden
    assert not re.search(r"ERROR DELEGATION_", output)


def test_git_operands_that_look_like_options_are_refused(world: dict[str, Any], tmp_path: Path) -> None:
    repo = world["repo"]
    for bad in ("--output=x", "-p", ""):
        with pytest.raises(ValueError):
            iiac_verifier_common.run_git(repo, "show", operands=(bad,))
    with pytest.raises(ValueError):
        iiac_verifier_common.show_blob(repo, "--output=x", "a")
    with pytest.raises(ValueError):
        iiac_verifier_common.is_ancestor_or_equal(repo, "--all", "HEAD")
    with pytest.raises(ValueError):
        iiac_verifier_common.run_git(repo, "show", options=("HEAD",))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(_manifest({}, world["tokens"])), encoding="utf-8")
    rc, output = _run(verify_iiac_manifest_artifacts.main, [str(manifest), "--repo", str(repo), "--at=--output=evil"])
    assert rc == 2, output
    assert not (repo / "evil").exists()
    ticket = next(iter(world["tickets"].values()))
    rc, output = _run(verify_iiac_delegation_refs.main, [str(ticket), "--repo", str(repo), "--approval-commit=--all"])
    assert rc == 2, output
