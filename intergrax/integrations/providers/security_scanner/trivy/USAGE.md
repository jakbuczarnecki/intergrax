# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `trivy` integration — usage

**Category:** `security_scanner`  
**Catalog factory:** ``create_trivy_security_scanner()``  
**Env prefix:** ``INTERGRAX_TRIVY_*``

```python
from intergrax.integrations.providers.security_scanner.trivy.bundle import create_trivy_security_scanner

backend = create_trivy_security_scanner()
```
