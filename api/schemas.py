from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    Recency: int = Field(..., ge=0)
    Frequency: int = Field(..., ge=0)
    Monetary_Total: float = Field(..., ge=0)
    Monetary_Mean: float = Field(..., ge=0)
    Monetary_Std: float = Field(..., ge=0)
    ChannelId: str = Field(...)


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool
