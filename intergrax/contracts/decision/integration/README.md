# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# Decision System Integration Boundary

Maps reference Decision System artifacts to platform decision contracts through
swappable adapter plugins. The integration **engine** depends only on protocols;
the **platform composition root** (`intergrax/runtime/decision_integration_composition.py`)
wires default and custom plugins.

## Composition flow

```text
Platform (decision_plugin_composition)

        |

Decision Integration Composition Root

        |

DecisionSystemIntegrationFactory

        |

DecisionSystemIntegrationEngine (contract-based)

        |

DecisionIntegrationAdapterProvider[]  →  Adapters
```

Entry from the Decision plugin composition module:

```text
compose_decision_system_integration_from_platform()
        → compose_decision_system_integration_engine()
        → DecisionSystemIntegrationFactory.create_engine()
```

## Forbidden practices

- Do not import concrete adapters inside `engine.py`.
- Do not use `getattr`, `setattr`, or `dict[str, Any]` for wiring.
- Do not introduce global registries, singleton engines, or hidden auto-discovery.
- Do not couple this boundary to reference matrix execution or `intergrax/runtime/execution`.

## Dependency injection

Composition root creates plugins and injects them into the factory-built engine.
The engine must not construct its own adapter implementations.
