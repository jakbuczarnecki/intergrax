# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1-R2 adversarial static version resolution (AST-only, no runtime import)."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_ast_signals import (
    extract_class_version_fields,
    parse_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def _fields(source: str) -> list:
    tree = parse_module("synthetic/r1_static_version_probe.py", source)
    return extract_class_version_fields("synthetic/r1_static_version_probe.py", tree)


def test_cx_r1_r2_static_a_direct_literal() -> None:
    source = '''
class Contract:
    schema_version = "foo.v1"
'''
    signals = _fields(source)
    assert len(signals) == 1
    assert signals[0].version_literal == "foo.v1"
    assert signals[0].static_version_conflict is False


def test_cx_r1_r2_static_b_same_module_constant() -> None:
    source = '''
FOO_SCHEMA_VERSION = "foo.v1"

class Contract:
    schema_version = FOO_SCHEMA_VERSION
'''
    signals = _fields(source)
    assert len(signals) == 1
    assert signals[0].version_literal == "foo.v1"
    assert signals[0].resolved_via_constant == "FOO_SCHEMA_VERSION"


def test_cx_r1_r2_static_c_annotated_constant_and_literal() -> None:
    source = '''
from typing import Final, Literal

FOO_SCHEMA_VERSION: Final[str] = "foo.v1"

class Contract:
    schema_version: Literal["foo.v1"] = FOO_SCHEMA_VERSION
'''
    signals = _fields(source)
    assert len(signals) == 1
    assert signals[0].version_literal == "foo.v1"
    assert signals[0].resolved_via_constant == "FOO_SCHEMA_VERSION"


def test_cx_r1_r2_static_d_literal_annotation_unresolved_default() -> None:
    source = '''
from typing import Literal

class Contract:
    schema_version: Literal["foo.v1"]
'''
    tree = parse_module("synthetic/r1_static_version_probe.py", source)
    signals = extract_class_version_fields("synthetic/r1_static_version_probe.py", tree)
    assert len(signals) == 1
    assert signals[0].version_literal == "foo.v1"


def test_cx_r1_r2_static_e_conflicting_annotation_and_default() -> None:
    source = '''
from typing import Literal

FOO_SCHEMA_VERSION = "foo.v2"

class Contract:
    schema_version: Literal["foo.v1"] = FOO_SCHEMA_VERSION
'''
    signals = _fields(source)
    assert len(signals) == 1
    assert signals[0].version_literal is None
    assert signals[0].static_version_conflict is True


def test_cx_r1_r2_static_f_unknown_name() -> None:
    source = '''
class Contract:
    schema_version = SOME_DYNAMIC_VALUE
'''
    signals = _fields(source)
    assert len(signals) == 1
    assert signals[0].version_literal is None
    assert signals[0].unresolved_static_reference is True


def test_cx_r1_r2_static_g_function_call_unresolved() -> None:
    source = '''
class Contract:
    schema_version = get_version()
'''
    signals = _fields(source)
    assert len(signals) == 1
    assert signals[0].version_literal is None
    assert signals[0].unresolved_static_reference is True
