# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `azure_key_vault` integration — usage

**Category:** `secrets_store`  
**Catalog factory:** ``create_azure_key_vault_secrets_store()``  
**Env prefix:** ``INTERGRAX_AZURE_KEY_VAULT_*``

```python
from intergrax.integrations.providers.secrets_store.azure_key_vault.bundle import create_azure_key_vault_secrets_store

backend = create_azure_key_vault_secrets_store()
```
