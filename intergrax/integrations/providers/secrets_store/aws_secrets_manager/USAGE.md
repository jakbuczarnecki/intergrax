# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `aws_secrets_manager` integration — usage

**Category:** `secrets_store`  
**Catalog factory:** ``create_aws_secrets_manager_secrets_store()``  
**Env prefix:** ``INTERGRAX_AWS_SECRETS_MANAGER_*``

```python
from intergrax.integrations.providers.secrets_store.aws_secrets_manager.bundle import create_aws_secrets_manager_secrets_store

backend = create_aws_secrets_manager_secrets_store()
```
