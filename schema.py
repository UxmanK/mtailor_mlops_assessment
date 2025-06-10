from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

# Response models
class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    class_id: int = Field(
        ...,
        description="The predicted class ID from ImageNet (0-999)",
        example=0,
        ge=0,
        lt=1000
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the prediction",
        example=0.9876
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the image in seconds",
        example=0.123,
        ge=0.0
    )

    class Config:
        schema_extra = {
            "example": {
                "class_id": 0,
                "confidence": 0.9876,
                "processing_time": 0.123
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(
        ...,
        description="Current status of the API",
        example="healthy"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is successfully loaded",
        example=True
    )
    version: str = Field(
        ...,
        description="Current version of the API",
        example="1.0.0"
    )
    last_prediction: Optional[Dict[str, Any]] = Field(
        None,
        description="Details of the last successful prediction, if any"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "last_prediction": {
                    "timestamp": "2024-03-20T12:00:00",
                    "class_id": 0,
                    "confidence": 0.9876,
                    "processing_time": 0.123
                }
            }
        }
