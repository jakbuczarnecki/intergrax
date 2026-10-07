# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R8 qualification-only composition fixture."""

from agents.nonexistent_r8_fixture_module import Foo


def invoke() -> None:
    Foo()
