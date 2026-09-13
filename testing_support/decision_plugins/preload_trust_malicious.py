# © Artur Czarnecki. All rights reserved.

"""Test-only module that must never be imported before admission (P0-A-R2)."""

raise RuntimeError("should never import")
