# `vault` integration — usage

**Category:** ``secrets_store``  
**Catalog factory:** ``create_vault_secrets_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(secrets_store=IntegrationSlug.VAULT)
backend = profile.resolve(IntegrationCategory.SECRETS_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store

backend = create_vault_secrets_store(**config_overrides)
```


## Environment variables

`INTERGRAX_VAULT_ADDR`, `INTERGRAX_VAULT_TOKEN`, `INTERGRAX_VAULT_MOUNT`

## Example

```python
from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store

secrets = create_vault_secrets_store(addr="http://127.0.0.1:8200", token="...")
secrets.put_secret("tenant/openai", "sk-...")
```

## Notes

HashiCorp Vault KV v2. Requires ``hvac``. New category ``secrets_store`` (§5.2.4).
