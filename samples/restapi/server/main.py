import joblib
from fastapi import FastAPI

from classifier import IrisClassifier, DummyIrisClassifier
from models import IrisFeatures, PredictionResponse

app = FastAPI(title="Iris Classification API")

# Load model from disk and initialize the classifier; or use a dummy classifier for testing.
# Be sure to set the working directory to the server folder so that the relative path to the model is correct.
LOAD_MODEL = False
classifier = None
if LOAD_MODEL:
    model = joblib.load("../models/iris_model.joblib")
    classifier = IrisClassifier(model)
else:
    classifier = DummyIrisClassifier()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """Endpoint to predict the Iris species based on input features."""

    # Since the classifier's predict method already returns a PredictionResponse, we can directly return it.
    return classifier.predict(features)
