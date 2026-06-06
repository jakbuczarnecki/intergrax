# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `grafana_oncall` integration — usage

**Category:** `notification_channel`  
**Catalog factory:** ``create_grafana_oncall_notification_channel()``  
**Env prefix:** ``INTERGRAX_GRAFANA_ONCALL_*``

```python
from intergrax.integrations.providers.notification_channel.grafana_oncall.bundle import create_grafana_oncall_notification_channel

backend = create_grafana_oncall_notification_channel()
```
