# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W3-R1 — typed contract and reflection gates for W3 seams."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.persisted_run_trace import (
    PersistedRunErrorCode,
    PersistedTraceEvent,
    RunError,
    RunStats,
    decode_persisted_run_error,
    decode_persisted_run_stats,
    decode_persisted_trace_event,
    parse_persisted_run_error_code,
    persisted_trace_event_to_wire,
    run_error_to_storage_dict,
    run_stats_to_storage_dict,
)
from intergrax.contracts.structured_json_value import StructuredJsonObject

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_W3_TYPED_SEAM_FILES = (
    _REPO_ROOT / "intergrax" / "contracts" / "persisted_run_trace.py",
    _REPO_ROOT / "intergrax" / "runtime" / "architecture" / "retrieval_security.py",
    _REPO_ROOT / "intergrax" / "websearch" / "contracts" / "routing_snapshot_sync.py",
)

_W3_REFLECTION_PATHS = (
    _REPO_ROOT / "intergrax" / "contracts" / "persisted_run_trace.py",
    _REPO_ROOT / "intergrax" / "runtime" / "architecture" / "retrieval_security.py",
    _REPO_ROOT / "intergrax" / "runtime" / "architecture" / "retrieval_security_wiring.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tracing" / "sqlite_run_trace_store.py",
    _REPO_ROOT / "intergrax" / "websearch" / "contracts" / "routing_snapshot_sync.py",
)


def _annotation_uses_forbidden(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in {"object", "Any"}
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "typing" and node.attr in {"Any", "Optional"}:
            return node.attr == "Any"
    if isinstance(node, ast.Subscript):
        return _annotation_uses_forbidden(node.value) or _annotation_uses_forbidden(node.slice)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_uses_forbidden(node.left) or _annotation_uses_forbidden(node.right)
    if isinstance(node, ast.Tuple):
        return any(_annotation_uses_forbidden(elt) for elt in node.elts)
    return False


def _resolve_type_aliases(tree: ast.Module) -> dict[str, ast.AST]:
    aliases: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            aliases[node.target.id] = node.value
    return aliases


def _expand_annotation(node: ast.AST, aliases: dict[str, ast.AST], *, depth: int = 0) -> ast.AST:
    if depth > 8:
        return node
    if isinstance(node, ast.Name) and node.id in aliases:
        return _expand_annotation(aliases[node.id], aliases, depth=depth + 1)
    return node


def _collect_function_annotations(
    func: ast.FunctionDef,
    label_prefix: str,
    aliases: dict[str, ast.AST],
    collected: list[tuple[str, ast.AST]],
) -> None:
    if func.returns is not None:
        collected.append(
            (f"{label_prefix}.__returns__", _expand_annotation(func.returns, aliases))
        )
    all_args = (
        *func.args.posonlyargs,
        *func.args.args,
        *func.args.kwonlyargs,
    )
    for arg in all_args:
        if arg.annotation is not None:
            collected.append(
                (f"{label_prefix}.{arg.arg}", _expand_annotation(arg.annotation, aliases))
            )
    if func.args.vararg is not None and func.args.vararg.annotation is not None:
        collected.append(
            (
                f"{label_prefix}.*{func.args.vararg.arg}",
                _expand_annotation(func.args.vararg.annotation, aliases),
            )
        )
    if func.args.kwarg is not None and func.args.kwarg.annotation is not None:
        collected.append(
            (
                f"{label_prefix}.**{func.args.kwarg.arg}",
                _expand_annotation(func.args.kwarg.annotation, aliases),
            )
        )


def _collect_public_annotations(path: Path) -> list[tuple[str, ast.AST]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _resolve_type_aliases(tree)
    collected: list[tuple[str, ast.AST]] = []
    for alias_name, rhs in aliases.items():
        collected.append((f"alias:{alias_name}", _expand_annotation(rhs, aliases)))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.annotation is not None:
            collected.append((node.target.id, _expand_annotation(node.annotation, aliases)))
        if isinstance(node, ast.FunctionDef):
            _collect_function_annotations(node, node.name, aliases, collected)
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.annotation is not None:
                collected.append(
                    (
                        f"{node.name}.{item.target.id}",
                        _expand_annotation(item.annotation, aliases),
                    )
                )
            if isinstance(item, ast.FunctionDef):
                _collect_function_annotations(
                    item,
                    f"{node.name}.{item.name}",
                    aliases,
                    collected,
                )
    return collected


def _reflection_compat_hits(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "setattr", "hasattr"}:
                hits.append(node.func.id)
    return hits


@pytest.mark.parametrize("path", _W3_TYPED_SEAM_FILES, ids=lambda p: p.name)
def test_harness_01_w3_r1_seams_forbid_object_and_any_annotations(path: Path) -> None:
    offenders = [
        label
        for label, annotation in _collect_public_annotations(path)
        if _annotation_uses_forbidden(annotation)
    ]
    assert not offenders, f"{path.name} forbidden annotations: {offenders}"


def test_harness_01_w3_r1_negative_synthetic_mapping_object_fails_gate() -> None:
    tree = ast.parse("payload: Mapping[str, object]\n")
    node = tree.body[0]
    assert isinstance(node, ast.AnnAssign) and node.annotation is not None
    assert _annotation_uses_forbidden(node.annotation)


def test_harness_01_w3_r1_negative_synthetic_dict_any_fails_gate() -> None:
    tree = ast.parse("payload: dict[str, Any]\n")
    node = tree.body[0]
    assert isinstance(node, ast.AnnAssign) and node.annotation is not None
    assert _annotation_uses_forbidden(node.annotation)


def test_harness_01_w3_r1_negative_synthetic_nested_mapping_object_fails_gate() -> None:
    tree = ast.parse("payload: Mapping[str, list[object]]\n")
    node = tree.body[0]
    assert isinstance(node, ast.AnnAssign) and node.annotation is not None
    assert _annotation_uses_forbidden(node.annotation)


def test_harness_01_w3_r1_negative_synthetic_alias_mapping_object_fails_gate() -> None:
    source = "RawPayload = Mapping[str, object]\n"
    tree = ast.parse(source)
    aliases = _resolve_type_aliases(tree)
    assert "RawPayload" in aliases
    assert _annotation_uses_forbidden(_expand_annotation(aliases["RawPayload"], aliases))


def test_harness_01_w3_r1_negative_synthetic_recursive_alias_fails_gate() -> None:
    source = "AliasB = Mapping[str, object]\nAliasA = AliasB\n"
    tree = ast.parse(source)
    aliases = _resolve_type_aliases(tree)
    assert _annotation_uses_forbidden(_expand_annotation(aliases["AliasA"], aliases))


def test_harness_01_w3_r1_negative_synthetic_function_arg_alias_fails_gate() -> None:
    source = "RawPayload = Mapping[str, object]\ndef decode(raw: RawPayload) -> None: ...\n"
    tree = ast.parse(source)
    offenders = [
        label
        for label, annotation in _collect_public_annotations_from_tree(tree)
        if _annotation_uses_forbidden(annotation)
    ]
    assert "decode.raw" in offenders


def test_harness_01_w3_r1_negative_synthetic_function_return_alias_fails_gate() -> None:
    source = "RawPayload = Mapping[str, object]\ndef decode() -> RawPayload: ...\n"
    tree = ast.parse(source)
    offenders = [
        label
        for label, annotation in _collect_public_annotations_from_tree(tree)
        if _annotation_uses_forbidden(annotation)
    ]
    assert "decode.__returns__" in offenders


def _collect_public_annotations_from_tree(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    aliases = _resolve_type_aliases(tree)
    collected: list[tuple[str, ast.AST]] = []
    for alias_name, rhs in aliases.items():
        collected.append((f"alias:{alias_name}", _expand_annotation(rhs, aliases)))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            _collect_function_annotations(node, node.name, aliases, collected)
    return collected


def test_harness_01_w3_r1_decoder_boundaries_use_structured_json_object() -> None:
    path = _REPO_ROOT / "intergrax" / "contracts" / "persisted_run_trace.py"
    decode_labels = {
        label
        for label, _ in _collect_public_annotations(path)
        if label.startswith("decode_persisted_")
    }
    assert decode_labels >= {
        "decode_persisted_run_stats.raw",
        "decode_persisted_run_error.raw",
        "decode_persisted_trace_event.raw",
    }
    offenders = [
        label
        for label, annotation in _collect_public_annotations(path)
        if label.endswith(".raw") and label.startswith("decode_persisted_")
        and _annotation_uses_forbidden(annotation)
    ]
    assert offenders == []


def test_harness_01_w3_r1_positive_typed_persisted_models_pass_gate() -> None:
    usage: StructuredJsonObject = {"total_tokens": 3}
    stats = RunStats(duration_ms=1, llm_usage=usage)
    assert run_stats_to_storage_dict(stats)["llm_usage"] == usage


@pytest.mark.parametrize("path", _W3_REFLECTION_PATHS, ids=lambda p: p.relative_to(_REPO_ROOT).as_posix())
def test_harness_01_w3_r1_no_reflection_compatibility(path: Path) -> None:
    hits = _reflection_compat_hits(path)
    assert hits == [], f"{path} reflection compat: {hits}"


def test_persisted_trace_roundtrip_and_legacy_error_code() -> None:
    event = PersistedTraceEvent(
        event_id="e1",
        run_id="r1",
        seq=1,
        ts_utc="2026-01-01T00:00:00Z",
        level="INFO",
        component="engine",
        step="plan",
        message="ok",
        payload={"k": "v"},
        tags={"tenant": "t1"},
    )
    wire = persisted_trace_event_to_wire(event)
    decoded = decode_persisted_trace_event(wire)
    assert decoded == event

    assert parse_persisted_run_error_code("internal_error") is PersistedRunErrorCode.INTERNAL_ERROR
    assert parse_persisted_run_error_code("WorkerFailed") is PersistedRunErrorCode.UNKNOWN

    error = decode_persisted_run_error({"error_type": "tool_error", "message": "x"})
    assert run_error_to_storage_dict(error) == {"error_type": "tool_error", "message": "x"}


def test_retrieval_poisoning_protocol_exposes_source_ref_only() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "architecture" / "retrieval_security.py"
    source = path.read_text(encoding="utf-8")
    assert "def metadata" not in source
    assert "source_ref" in source
