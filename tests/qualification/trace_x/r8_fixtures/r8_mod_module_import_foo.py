# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R8 qualification-only composition fixture."""

import tests.qualification.trace_x.r8_fixtures.r8_package_module_defines_foo as r8_pkg


def invoke() -> None:
    r8_pkg.Foo()
