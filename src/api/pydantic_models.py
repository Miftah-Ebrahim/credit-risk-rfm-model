from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    Recency: int = Field(..., description="Days since last transaction", ge=0)
    Frequency: int = Field(..., description="Total number of transactions", ge=0)
    Monetary_Total: float = Field(..., description="Total transaction amount", ge=0)
    Monetary_Mean: float = Field(..., description="Average transaction amount", ge=0)
    Monetary_Std: float = Field(
        ..., description="Standard deviation of transaction amounts", ge=0
    )


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool
    status: str = "success"
