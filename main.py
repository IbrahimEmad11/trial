from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("rf_lymphedema_model.joblib")

# Define input schema
class PatientFeatures(BaseModel):
    Age: float
    Menopausal_status: int
    BMI: float
    pastDM: int
    pastHypertension: int
    pastCardiac: int
    pastLiver: int
    pastRenalproblems: int
    pastScrewsandplatel: int
    Laterality: int
    T: int
    N: int
    M: int
    Specimen_type: int
    Lymph_node_1: int
    Lymph_node: int
    Peritumoural_lymphovascular_invasion: int
    chemotherapy: int
    Radiotherapy: int
    Hormanal: int
    Pain: int
    Tenderness: int
    Stiffness: int
    Weakness: int
    Referralpain: int
    Swelling: int

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Lymphedema Prediction API"}

@app.post("/predict")
def predict(features: PatientFeatures):
    # Convert to numpy array
    input_data = np.array([[getattr(features, col) for col in features.__annotations__]])

    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of class 1

    return {
        "prediction": int(prediction),
        "probability_of_lymphedema": round(probability, 4)
    }
