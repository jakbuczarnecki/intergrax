#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LCI-1D — Native KnowledgeDocument conformance gate (AST, public ABI, isolated runtime proof)."""

from __future__ import annotations

import argparse
import ast
import inspect
import subprocess
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_SCAN_ROOT = "intergrax/knowledge"

EXPECTED_PUBLIC_EXPORTS: tuple[str, ...] = (
    "KnowledgeDocument",
    "KnowledgeDocumentIdentity",
    "KnowledgeDocumentProvenance",
    "KnowledgeDocumentScope",
    "dump_knowledge_document",
    "load_knowledge_document",
)

EXPECTED_MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    "KnowledgeDocument": (
        "schema_version",
        "identity",
        "scope",
        "content",
        "metadata",
        "provenance",
    ),
    "KnowledgeDocumentIdentity": (
        "document_id",
        "root_document_id",
        "parent_document_id",
    ),
    "KnowledgeDocumentScope": (
        "tenant_id",
        "namespace",
    ),
    "KnowledgeDocumentProvenance": (
        "source_kind",
        "source_id",
        "source_parent_id",
        "provider_id",
        "source_revision",
        "source_uri",
        "content_hash",
    ),
}

EXPECTED_SERIALIZER_PARAMS: dict[str, tuple[str, ...]] = {
    "dump_knowledge_document": ("document",),
    "load_knowledge_document": ("payload",),
}

BLOCKED_MODULE_PREFIXES: tuple[str, ...] = (
    "langchain",
    "langgraph",
    "intergrax.compat",
)


def is_blocked_module(module: str) -> bool:
    if module == "langchain" or module.startswith("langchain_"):
        return True
    if module == "langgraph" or module.startswith("langgraph_"):
        return True
    if module == "intergrax.compat" or module.startswith("intergrax.compat."):
        return True
    return False


def _literal_module(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_importlib_module(call: ast.Call) -> str | None:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "importlib":
        return None
    if func.attr != "import_module" or not call.args:
        return None
    return _literal_module(call.args[0])


def _extract_dunder_import_module(call: ast.Call) -> str | None:
    func = call.func
    if not isinstance(func, ast.Name) or func.id != "__import__":
        return None
    if not call.args:
        return None
    return _literal_module(call.args[0])


def _is_type_checking_test(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "TYPE_CHECKING"


class _KnowledgeImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []
        self._type_checking_depth = 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            self._type_checking_depth += 1
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)
            self._type_checking_depth -= 1
            return
        self.generic_visit(node)

    def _record(self, lineno: int, description: str) -> None:
        if self._type_checking_depth > 0:
            return
        self.violations.append((lineno, description))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_blocked_module(alias.name):
                self._record(
                    node.lineno,
                    f"forbidden import '{alias.name}'",
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and is_blocked_module(node.module):
            names = ", ".join(sorted(alias.name for alias in node.names))
            self._record(
                node.lineno,
                f"forbidden from-import '{node.module}' ({names})",
            )

    def visit_Call(self, node: ast.Call) -> None:
        module = _extract_importlib_module(node)
        if module and is_blocked_module(module):
            self._record(
                node.lineno,
                f"forbidden importlib.import_module('{module}')",
            )
        module = _extract_dunder_import_module(node)
        if module and is_blocked_module(module):
            self._record(
                node.lineno,
                f"forbidden __import__('{module}')",
            )
        self.generic_visit(node)


def iter_knowledge_python_files(repo_root: Path) -> Iterator[Path]:
    root = repo_root / KNOWLEDGE_SCAN_ROOT
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def scan_ast_boundary(repo_root: Path) -> list[str]:
    diagnostics: list[str] = []
    for path in iter_knowledge_python_files(repo_root):
        rel_path = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            diagnostics.append(
                f"KNOWLEDGE_DOCUMENT_AST_VIOLATION:{rel_path}:{exc.lineno or 0}:syntax error: {exc.msg}"
            )
            continue
        visitor = _KnowledgeImportVisitor()
        visitor.visit(tree)
        for lineno, description in sorted(visitor.violations, key=lambda item: (item[0], item[1])):
            diagnostics.append(
                f"KNOWLEDGE_DOCUMENT_AST_VIOLATION:{rel_path}:{lineno}:{description}"
            )
    return sorted(diagnostics)


def _public_surface_violation(description: str) -> str:
    return f"KNOWLEDGE_DOCUMENT_PUBLIC_SURFACE_VIOLATION:{description}"


def _isolated_import_failure(description: str) -> str:
    return f"KNOWLEDGE_DOCUMENT_ISOLATED_IMPORT_FAILURE:{description}"


def check_public_surface() -> list[str]:
    from intergrax.knowledge import contracts

    violations: list[str] = []

    actual_exports = tuple(sorted(contracts.__all__))
    expected_exports = tuple(sorted(EXPECTED_PUBLIC_EXPORTS))
    if actual_exports != expected_exports:
        violations.append(
            _public_surface_violation(
                f"__all__ mismatch: expected {expected_exports!r}, got {actual_exports!r}"
            )
        )

    models = {
        "KnowledgeDocument": contracts.KnowledgeDocument,
        "KnowledgeDocumentIdentity": contracts.KnowledgeDocumentIdentity,
        "KnowledgeDocumentScope": contracts.KnowledgeDocumentScope,
        "KnowledgeDocumentProvenance": contracts.KnowledgeDocumentProvenance,
    }

    for model_name, model in models.items():
        field_names = tuple(model.model_fields.keys())
        expected_fields = EXPECTED_MODEL_FIELDS[model_name]
        if field_names != expected_fields:
            violations.append(
                _public_surface_violation(
                    f"{model_name} fields mismatch: expected {expected_fields!r}, got {field_names!r}"
                )
            )
        config = model.model_config
        if config.get("extra") != "forbid":
            violations.append(
                _public_surface_violation(
                    f"{model_name} model_config extra must be 'forbid', got {config.get('extra')!r}"
                )
            )
        if config.get("frozen") is not True:
            violations.append(
                _public_surface_violation(
                    f"{model_name} model_config frozen must be True, got {config.get('frozen')!r}"
                )
            )

    for func_name, expected_params in EXPECTED_SERIALIZER_PARAMS.items():
        func = getattr(contracts, func_name)
        params = tuple(
            name
            for name, parameter in inspect.signature(func).parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if params != expected_params:
            violations.append(
                _public_surface_violation(
                    f"{func_name} parameters mismatch: expected {expected_params!r}, got {params!r}"
                )
            )

    return violations


def _isolated_proof_source(repo_root: Path) -> str:
    repo = str(repo_root).replace("\\", "/")
    return textwrap.dedent(
        f"""
        import math
        import sys
        from importlib.abc import MetaPathFinder
        from importlib.machinery import ModuleSpec

        REPO_ROOT = {repo!r}
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)

        BLOCKED_PREFIXES = ("langchain", "langgraph", "intergrax.compat")

        def _is_blocked(name: str) -> bool:
            if name == "langchain" or name.startswith("langchain_"):
                return True
            if name == "langgraph" or name.startswith("langgraph_"):
                return True
            if name == "intergrax.compat" or name.startswith("intergrax.compat."):
                return True
            return False

        class _BlockedImportFinder(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if _is_blocked(fullname):
                    raise ModuleNotFoundError(
                        f"blocked import namespace: {{fullname}}"
                    )
                return None

        for name in list(sys.modules):
            if _is_blocked(name):
                del sys.modules[name]

        sys.meta_path.insert(0, _BlockedImportFinder())

        from intergrax.knowledge.contracts import (
            KnowledgeDocument,
            dump_knowledge_document,
            load_knowledge_document,
        )

        def _identity(**overrides):
            payload = {{
                "document_id": "file:abc123",
                "root_document_id": "file:abc123",
                "parent_document_id": None,
            }}
            payload.update(overrides)
            return payload

        def _scope(**overrides):
            payload = {{"tenant_id": "tenant-1", "namespace": None}}
            payload.update(overrides)
            return payload

        def _provenance(**overrides):
            payload = {{
                "source_kind": "file",
                "source_id": "abc123",
                "source_parent_id": None,
                "provider_id": None,
                "source_revision": None,
                "source_uri": None,
                "content_hash": None,
            }}
            payload.update(overrides)
            return payload

        def _document(**overrides):
            payload = {{
                "schema_version": 1,
                "identity": _identity(),
                "scope": _scope(),
                "content": "Hello knowledge",
                "metadata": {{}},
                "provenance": _provenance(),
            }}
            payload.update(overrides)
            return KnowledgeDocument.model_validate(payload)

        def _document_omitting_metadata(**overrides):
            payload = {{
                "schema_version": 1,
                "identity": _identity(),
                "scope": _scope(),
                "content": "Hello knowledge",
                "provenance": _provenance(),
            }}
            payload.update(overrides)
            return KnowledgeDocument.model_validate(payload)

        def _expect_raises(exc_type, fn):
            try:
                fn()
            except exc_type:
                return
            except Exception as exc:
                raise AssertionError(f"expected {{exc_type.__name__}}, got {{type(exc).__name__}}") from exc
            raise AssertionError(f"expected {{exc_type.__name__}}, no exception raised")

        source = _document(content=" padded content ")
        assert source.identity.parent_document_id is None
        assert source.identity.root_document_id == source.identity.document_id
        assert source.content == " padded content "
        omitted = _document_omitting_metadata()
        _expect_raises(TypeError, lambda: omitted.metadata.__setitem__("k", 1))

        derivative = _document(
            identity=_identity(
                document_id="file:abc123:chunk:1",
                root_document_id="file:abc123",
                parent_document_id="file:abc123",
            )
        )
        assert derivative.identity.root_document_id == "file:abc123"
        assert derivative.identity.parent_document_id == "file:abc123"
        assert derivative.identity.parent_document_id != derivative.identity.document_id
        dumped_derivative = dump_knowledge_document(derivative)
        restored_derivative = load_knowledge_document(dumped_derivative)
        assert restored_derivative == derivative

        unicode_doc = _document(content="Unicode: łódź — π")
        dumped_once = dump_knowledge_document(unicode_doc)
        dumped_twice = dump_knowledge_document(unicode_doc)
        assert isinstance(dumped_once, bytes)
        assert dumped_once == dumped_twice
        restored_unicode = load_knowledge_document(dumped_once)
        assert restored_unicode == unicode_doc

        _expect_raises(
            Exception,
            lambda: _document(content=b"bytes"),
        )
        _expect_raises(
            Exception,
            lambda: KnowledgeDocument.model_validate({{
                "schema_version": True,
                "identity": _identity(),
                "scope": _scope(),
                "content": "x",
                "metadata": {{}},
                "provenance": _provenance(),
            }}),
        )
        _expect_raises(
            Exception,
            lambda: KnowledgeDocument.model_validate({{
                "schema_version": "1",
                "identity": _identity(),
                "scope": _scope(),
                "content": "x",
                "metadata": {{}},
                "provenance": _provenance(),
            }}),
        )
        _expect_raises(
            Exception,
            lambda: KnowledgeDocument.model_validate({{
                "schema_version": 1,
                "identity": _identity(),
                "scope": {{"namespace": "ns"}},
                "content": "x",
                "metadata": {{}},
                "provenance": _provenance(),
            }}),
        )
        _expect_raises(
            Exception,
            lambda: _document(content="   "),
        )

        meta_doc = _document(metadata={{"nested": {{"items": [1]}}}})
        _expect_raises(TypeError, lambda: meta_doc.metadata.__setitem__("k", 1))
        _expect_raises(
            TypeError,
            lambda: meta_doc.metadata["nested"].__setitem__("k", 1),
        )
        _expect_raises(
            TypeError,
            lambda: meta_doc.metadata["nested"]["items"].append(2),
        )
        _expect_raises(Exception, lambda: _document(metadata={{"tenant_id": "x"}}))
        _expect_raises(Exception, lambda: _document(metadata={{"nested": {{"api_key": "x"}}}}))
        _expect_raises(Exception, lambda: _document(metadata={{"value": math.nan}}))
        _expect_raises(Exception, lambda: _document(metadata={{"value": math.inf}}))
        _expect_raises(Exception, lambda: _document(metadata={{"value": b"bytes"}}))

        _expect_raises(ValueError, lambda: load_knowledge_document('{{"schema_version":1,"schema_version":2}}'))
        _expect_raises(ValueError, lambda: load_knowledge_document("[]"))
        _expect_raises(ValueError, lambda: load_knowledge_document(b"\\xff\\xfe"))
        _expect_raises(
            ValueError,
            lambda: load_knowledge_document(
                dump_knowledge_document(_document()).decode("utf-8").replace(
                    '"schema_version":1', '"schema_version":99'
                )
            ),
        )

        for name in list(sys.modules):
            if _is_blocked(name):
                raise AssertionError(f"blocked module loaded during proof: {{name}}")
        """
    ).strip()


def run_isolated_proof(repo_root: Path) -> list[str]:
    source = _isolated_proof_source(repo_root)
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown isolated proof failure").strip()
        return [_isolated_import_failure(detail)]
    return []


def audit_repository(repo_root: Path) -> list[str]:
    violations: list[str] = []
    violations.extend(scan_ast_boundary(repo_root))
    violations.extend(check_public_surface())
    violations.extend(run_isolated_proof(repo_root))
    return sorted(violations)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: auto-detected)",
    )
    args = parser.parse_args(list(argv) if argv is not None else [])

    violations = audit_repository(args.repo_root.resolve())
    if violations:
        for message in violations:
            print(message)
        return 1

    print("Knowledge document conformance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
