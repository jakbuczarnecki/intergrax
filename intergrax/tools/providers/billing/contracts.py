# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class BillingRecordUsageInput(BaseModel):
    customer_id: str = Field(..., min_length=1)
    metric: str = Field(..., min_length=1)
    quantity: float = Field(..., ge=0.0)


class BillingRecordUsageOutput(BaseModel):
    event_id: str
    customer_id: str
    metric: str
    quantity: float
    recorded: bool = True


class BillingListUsageInput(BaseModel):
    customer_id: str = Field(..., min_length=1)
    limit: int = Field(default=50, ge=1, le=500)


class BillingMeterEventOutput(BaseModel):
    event_id: str
    customer_id: str = ""
    metric: str = ""
    quantity: float = 0.0


class BillingListUsageOutput(BaseModel):
    customer_id: str
    events: list[BillingMeterEventOutput] = Field(default_factory=list)
    total: int = 0
