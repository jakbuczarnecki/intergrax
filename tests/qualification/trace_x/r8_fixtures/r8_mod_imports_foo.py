# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R8 qualification-only composition fixture."""

from tests.qualification.trace_x.r8_fixtures.r8_mod_defines_foo import Foo


def invoke() -> None:
    Foo()
