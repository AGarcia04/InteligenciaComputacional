import random
from abc import ABC, abstractmethod

from models import IrisFeatures, PredictionResponse

species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}


class BaseIrisClassifier(ABC):
    """Abstract base class for Iris classifiers."""

    @abstractmethod
    def predict(self, features: IrisFeatures) -> PredictionResponse:
        """Run prediction based on the input features and return a PredictionResponse."""
        pass


class DummyIrisClassifier(BaseIrisClassifier):
    """A dummy classifier that returns random predictions for testing purposes."""

    def predict(self, features: IrisFeatures) -> PredictionResponse:
        # Dummy prediction logic (for testing purposes)
        dummy_prediction = random.choice(list(species_map.keys()))
        dummy_confidence = 0.9  # High confidence for the dummy prediction
        return PredictionResponse(prediction=dummy_prediction, species=species_map[dummy_prediction],
                                  confidence=dummy_confidence)


class IrisClassifier(BaseIrisClassifier):
    """A real classifier that uses a trained model to make predictions."""

    def __init__(self, model):
        self.model = model

    def predict(self, features: IrisFeatures) -> PredictionResponse:
        # Prepare data for the model
        X = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]

        # Make prediction
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0].max()

        return PredictionResponse(prediction=int(prediction), species=species_map[prediction],
                                  confidence=float(confidence))
