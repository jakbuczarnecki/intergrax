# `aws` integration — usage

**Category:** ``cloud_platform``  
**Catalog factory:** ``create_aws_cloud_platform()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(cloud_platform=IntegrationSlug.AWS)
backend = profile.resolve(IntegrationCategory.CLOUD_PLATFORM)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.aws.bundle import create_aws_cloud_platform

backend = create_aws_cloud_platform(**config_overrides)
```


## Environment variables

`INTERGRAX_AWS_REGION`, `INTERGRAX_AWS_PROFILE`; optional keys or `INTERGRAX_AWS_ROLE_ARN`

## Example

```python
from intergrax.integrations.providers.aws.bundle import create_aws_cloud_platform

platform = create_aws_cloud_platform(region="eu-central-1")
health = platform.health()
s3_slug = platform.resolve("object_storage")  # -> "s3"
```

## Notes

boto3 SDK only in ``opens.py``. The facade does not implement S3/SQS — it returns default category slugs.
