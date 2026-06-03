# `lab_json` integration — usage

**Category:** ``interaction_surface``  
**Catalog factory:** ``create_lab_json_interaction_surface()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(interaction_surface="lab_json")
backend = profile.resolve(IntegrationCategory.INTERACTION_SURFACE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.interaction_surface.lab_json.bundle import create_lab_json_interaction_surface

backend = create_lab_json_interaction_surface(**config_overrides)
```


## Environment variables

Optional `INTERGRAX_LAB_JSON_DEFAULT_SOURCE`

## Example

```python
from intergrax.integrations.providers.interaction_surface.lab_json.bundle import create_lab_json_interaction_surface

surface = create_lab_json_interaction_surface()
if surface.can_handle(inbound_payload):
    message = surface.to_inbound(inbound_payload)
    print(message.text, message.channel)
```

## Notes

JSON intake for the lab; channel ``lab``.
