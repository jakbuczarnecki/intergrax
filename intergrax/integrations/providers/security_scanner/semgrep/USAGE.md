# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `semgrep` integration — usage

**Category:** `security_scanner`  
**Catalog factory:** ``create_semgrep_security_scanner()``  
**Env prefix:** ``INTERGRAX_SEMGREP_*``

```python
from intergrax.integrations.providers.security_scanner.semgrep.bundle import create_semgrep_security_scanner

backend = create_semgrep_security_scanner()
```
