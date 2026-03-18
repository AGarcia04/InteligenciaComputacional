from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    """Data model for the input features of the Iris dataset."""
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")


class PredictionResponse(BaseModel):
    """Data model for the response of the prediction endpoint."""
    prediction: int = Field(..., description="Predicted class label (0: setosa, 1: versicolor, 2: virginica)")
    species: str = Field(..., description="Predicted species name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction (0 to 1)")
