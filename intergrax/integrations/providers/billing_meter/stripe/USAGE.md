# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `stripe` integration — usage

**Category:** `billing_meter`  
**Catalog factory:** ``create_stripe_billing_meter()``  
**Env prefix:** ``INTERGRAX_STRIPE_*``

```python
from intergrax.integrations.providers.billing_meter.stripe.bundle import create_stripe_billing_meter

backend = create_stripe_billing_meter()
```
