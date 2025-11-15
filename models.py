"""Type definitions for Cancer Prediction API"""
from typing import List, Optional, Union
from typing_extensions import TypedDict


class PredictionRequest(TypedDict):
    """Type definition for prediction request"""
    Age: Union[int, float]
    Gender: int
    BMI: Union[int, float]
    Smoking: int
    GeneticRisk: int
    PhysicalActivity: Union[int, float]
    AlcoholIntake: Union[int, float]
    CancerHistory: int


class ProbabilityDict(TypedDict):
    """Type definition for probability dictionary"""
    benign: float
    malignant: float


class PredictionResponse(TypedDict):
    """Type definition for prediction response"""
    prediction: int
    diagnosis: str
    probability: ProbabilityDict


class ErrorResponse(TypedDict):
    """Type definition for error response"""
    error: str
    required_fields: Optional[List[str]]
