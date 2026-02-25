from fastapi import FastAPI
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Dummy training data
hours = np.arange(1, 25).reshape(-1, 1)
energy_usage = np.array([
    120, 115, 110, 105, 100, 130, 180, 200, 220, 250, 300, 320,
    310, 290, 270, 260, 240, 230, 210, 190, 170, 160, 140, 130
])

model = LinearRegression()
model.fit(hours, energy_usage)

@app.get("/")
def read_root():
    return {"message": "GreenPulse AI Backend Running"}

@app.get("/predict")
def predict_energy():
    future_hours = np.arange(25, 49).reshape(-1, 1)
    predictions = model.predict(future_hours)
    return {"predicted_energy": predictions.tolist()}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
