# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Static raw-secret gate for product source (PDC-D6-R007).

A deterministic AST scan, not a secret manager. It flags:

* ``env_secret_literal_fallback``: an environment read (``os.environ.get``/``os.getenv``)
  whose variable name or assignment target is secret-like and whose default is a non-empty
  string literal (the shape of the removed ``harness-dev-signing-key`` fallback);
* ``credential_format_literal``: a string literal in a well-known credential format
  (PEM private key block, cloud/VCS/chat token prefixes, JWT-like triples);
* ``secret_named_literal``: a secret-named assignment, keyword or default argument bound to a
  high-entropy literal;
* ``known_forbidden_literal``: literals that must never appear in product source.

Findings carry no literal value, so reports and logs never echo a candidate secret.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

RULE_ENV_FALLBACK = "env_secret_literal_fallback"
RULE_CREDENTIAL_FORMAT = "credential_format_literal"
RULE_SECRET_NAMED_LITERAL = "secret_named_literal"
RULE_KNOWN_FORBIDDEN = "known_forbidden_literal"

# Built from parts so that this module does not contain the forbidden literal itself.
KNOWN_FORBIDDEN_LITERALS: frozenset[str] = frozenset({"-".join(("harness", "dev", "signing", "key"))})

_SECRET_NAME = re.compile(
    r"(secret|passw(or)?d|passwd|api_?key|apikey|private_?key|signing_?key|access_?key|"
    r"auth_?token|bearer|credential|token(?!s))",
    re.IGNORECASE,
)
_NON_SECRET_SUFFIX = re.compile(
    r"(_env|_ref|_path|_id|_ids|_name|_names|_header|_headers|_url|_uri|_field|_kind|_type|"
    r"_class|_fingerprint|_digest|_scope|_scopes|_count|_counts|_limit|_ttl|_budget|_usage|"
    r"_provider|_prefix|_label|_key_id|_algorithm|_format|_endpoint|_mode|_policy|_store|"
    r"_resolver|_enabled|_required|_present|_status|_reason|_code|_param|_params|_text|"
    r"_message|_hint|_help|_description|_doc|_file|_dir|_cap|_max|_min|_secs|_seconds|_ms)$",
    re.IGNORECASE,
)
_CREDENTIAL_FORMATS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)
_ENV_READERS = frozenset({"get", "getenv"})
_ENV_OWNERS = frozenset({"environ", "os", "env"})
_ENV_VAR_LIKE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
# snake_case / kebab-case / dotted identifiers are labels and reason codes, not secrets.
_IDENTIFIER_LIKE = re.compile(r"[a-z][a-z0-9]*(?:[_.:\-][a-z0-9]+)+")
_MIN_NAMED_LITERAL_LEN = 20
_MIN_ENTROPY_BITS = 3.2


@dataclass(frozen=True, slots=True)
class RawSecretFinding:
    """One gate finding. ``name`` is the identifier involved, never the literal value."""

    path: str
    line: int
    rule: str
    name: str

    def key(self) -> str:
        return f"{self.path}:{self.rule}:{self.name}"


def _entropy_bits(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _is_secret_name(name: str) -> bool:
    if not name or not _SECRET_NAME.search(name):
        return False
    return _NON_SECRET_SUFFIX.search(name) is None


def _target_names(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _target_names(item)
    elif isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            yield key.value


def _str_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _looks_like_named_secret(value: str) -> bool:
    if len(value) < _MIN_NAMED_LITERAL_LEN or " " in value or "\n" in value:
        return False
    if _ENV_VAR_LIKE.match(value) or value.startswith(("http://", "https://", "<", "$", "{")):
        return False
    if "/" in value or "\\" in value or _IDENTIFIER_LIKE.fullmatch(value):
        return False
    return _entropy_bits(value) >= _MIN_ENTROPY_BITS


def _env_read_default(call: ast.Call) -> tuple[str | None, str | None]:
    """Return ``(variable_name_literal, default_literal)`` for an env-read call."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _ENV_READERS:
        return None, None
    owner = func.value
    owner_name = owner.id if isinstance(owner, ast.Name) else owner.attr if isinstance(owner, ast.Attribute) else ""
    if owner_name not in _ENV_OWNERS:
        return None, None
    if func.attr == "getenv" and owner_name != "os":
        return None, None
    var = _str_literal(call.args[0]) if call.args else None
    default = _str_literal(call.args[1]) if len(call.args) > 1 else None
    if default is None:
        for keyword in call.keywords:
            if keyword.arg == "default":
                default = _str_literal(keyword.value)
    return var, default


def scan_source_for_raw_secrets(source: str, *, path: str) -> tuple[RawSecretFinding, ...]:
    """Scan one Python module's source text; syntax errors yield no findings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    findings: list[RawSecretFinding] = []

    def add(node: ast.AST, rule: str, name: str) -> None:
        findings.append(RawSecretFinding(path=path, line=getattr(node, "lineno", 0), rule=rule, name=name))

    assigned_call_targets: dict[int, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [n for target in node.targets for n in _target_names(target)]
            if isinstance(node.value, ast.Call):
                assigned_call_targets[id(node.value)] = names
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            assigned_call_targets[id(node.value)] = list(_target_names(node.target))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value in KNOWN_FORBIDDEN_LITERALS:
                add(node, RULE_KNOWN_FORBIDDEN, "literal")
            elif any(pattern.search(value) for pattern in _CREDENTIAL_FORMATS):
                add(node, RULE_CREDENTIAL_FORMAT, "literal")
        if isinstance(node, ast.Call):
            var, default = _env_read_default(node)
            if default:
                targets = assigned_call_targets.get(id(node), [])
                if (var and _SECRET_NAME.search(var)) or any(_is_secret_name(t) for t in targets):
                    add(node, RULE_ENV_FALLBACK, var or (targets[0] if targets else "env"))
            for keyword in node.keywords:
                literal = _str_literal(keyword.value)
                if (
                    keyword.arg
                    and literal is not None
                    and _is_secret_name(keyword.arg)
                    and _looks_like_named_secret(literal)
                ):
                    add(node, RULE_SECRET_NAMED_LITERAL, keyword.arg)
        elif isinstance(node, ast.Assign):
            literal = _str_literal(node.value)
            if literal is not None and _looks_like_named_secret(literal):
                for name in (n for t in node.targets for n in _target_names(t)):
                    if _is_secret_name(name):
                        add(node, RULE_SECRET_NAMED_LITERAL, name)
        elif isinstance(node, ast.AnnAssign):
            literal = _str_literal(node.value)
            if literal is not None and _looks_like_named_secret(literal):
                for name in _target_names(node.target):
                    if _is_secret_name(name):
                        add(node, RULE_SECRET_NAMED_LITERAL, name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
            pairs = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
            for arg, default_node in pairs:
                literal = _str_literal(default_node)
                if literal is not None and _is_secret_name(arg.arg) and _looks_like_named_secret(literal):
                    add(node, RULE_SECRET_NAMED_LITERAL, arg.arg)
    return tuple(findings)


def iter_python_files(
    root: Path,
    *,
    relative_to: Path,
    excluded_dir_names: frozenset[str] = frozenset(
        {"tests", "test", "__pycache__", ".venv", "build", "node_modules", ".tmp", "fixtures", "scripts"}
    ),
) -> Iterator[tuple[Path, str]]:
    """Yield ``(file, posix_relative_path)`` for product ``*.py`` files under ``root``."""
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(relative_to)
        if any(part in excluded_dir_names for part in relative.parts):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        yield path, relative.as_posix()


def scan_tree_for_raw_secrets(
    roots: Iterable[Path],
    *,
    relative_to: Path,
) -> tuple[RawSecretFinding, ...]:
    """Scan product source under ``roots``; returns findings sorted by path and line."""
    findings: list[RawSecretFinding] = []
    for root in roots:
        if not root.exists():
            continue
        for file_path, relative in iter_python_files(root, relative_to=relative_to):
            try:
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            findings.extend(scan_source_for_raw_secrets(text, path=relative))
    return tuple(sorted(findings, key=lambda item: (item.path, item.line, item.rule, item.name)))


__all__ = [
    "KNOWN_FORBIDDEN_LITERALS",
    "RULE_CREDENTIAL_FORMAT",
    "RULE_ENV_FALLBACK",
    "RULE_KNOWN_FORBIDDEN",
    "RULE_SECRET_NAMED_LITERAL",
    "RawSecretFinding",
    "iter_python_files",
    "scan_source_for_raw_secrets",
    "scan_tree_for_raw_secrets",
]
