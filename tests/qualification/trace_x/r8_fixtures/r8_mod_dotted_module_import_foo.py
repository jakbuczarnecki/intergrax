# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R8 qualification-only composition fixture."""

import tests.qualification.trace_x.r8_fixtures.r8_package_module_defines_foo


def invoke() -> None:
    tests.qualification.trace_x.r8_fixtures.r8_package_module_defines_foo.Foo()
