# © Artur Czarnecki. All rights reserved.

"""Application observability extensions for governed_contractor_application."""

from applications.governed_contractor_application.tracing.registry import register_tracing_schemas

__all__ = ["register_tracing_schemas"]
