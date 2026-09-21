# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Websearch tool bundle — register via ``intergrax.tools.registry.shipped_plugins``."""

# Avoid eager imports here: ``service`` depends on ``ToolWiringContext`` while registry
# wiring references ``executor_contract`` in the same package.

__all__: tuple[str, ...] = ()
