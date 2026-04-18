from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# ==========================================
# THE STRIVERS - E4C AI DIAGNOSTIC ENGINE
# ==========================================
# BASELINES DERIVED FROM E4C SOLUTIONS LIBRARY:
# - RainMaker 2: Max motor current 8.0 A, Voltage 24-34V, 50 L/min, 70m depth.
# - Shurflo 9300: 24 VDC operation, 82 gal/h at 230 ft.
# - SDS-T-128: 12-30 V DC, 4-inch casing.
# - GF1: 12/24 V DC for remote irrigation.
# ENGINEERING NOTE: E4C documentation lacks specific fault signatures 
# for sand clogging or dry running. The Strivers AI provides the 
# predictive troubleshooting layer to bridge this gap.

# FEATURES: ["Irradiance", "Current", "Temperature", "Vibration", "Flow", "Pressure"]
feature_names = ["Irradiance", "Current", "Temperature", "Vibration", "Flow", "Pressure"]

# ROBUST TRAINING (Based on E4C Standards)
np.random.seed(42)
# Baseline centered around RainMaker 2 nominal 8.0A
X_normal = np.random.normal(loc=[800, 8.0, 35, 1.5, 50, 30], scale=[10, 0.2, 1, 0.1, 1, 1], size=(100, 6))
X_dry = np.random.normal(loc=[800, 5.5, 65, 2.5, 5, 10], scale=[10, 0.3, 3, 0.3, 1, 2], size=(100, 6))
X_sand = np.random.normal(loc=[800, 11.5, 52, 3.5, 15, 45], scale=[10, 0.4, 2, 0.4, 2, 2], size=(100, 6))
X_cav = np.random.normal(loc=[800, 8.1, 40, 9.5, 40, 25], scale=[10, 0.2, 1, 1.0, 3, 2], size=(100, 6))

X_train = np.vstack([X_normal, X_dry, X_sand, X_cav])
y_train = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

class Telemetry(BaseModel):
    Irradiance: float; Current: float; Temp: float; Vibration: float; Flow: float; Head: float

@app.post("/predict")
def predict(data: Telemetry):
    raw = np.array([[data.Irradiance, data.Current, data.Temp, data.Vibration, data.Flow, data.Head]])
    pred = int(model.predict(raw)[0])
    conf = float(max(model.predict_proba(raw)[0])) * 100
    
    # E4C EXPERT SYSTEM KNOWLEDGE BASE
    diag = {
        0: {"status": "Healthy", "issue": "Normal Operation", "health": 100, "exp": "Telemetry signature is optimal. Current matches RainMaker 2 baseline (~8.0A).", "sop": "Maintain standard monthly visual inspection."},
        1: {"status": "Critical", "issue": "Dry Running", "health": int(100 - conf*0.8), "exp": f"Current ({data.Current}A) is abnormally low indicating loss of prime. Stator overheating risk.", "sop": "IMMEDIATE SHUTDOWN. Verify static well water level and low-water probe calibration."},
        2: {"status": "Critical", "issue": "Sand Clogging", "health": int(100 - conf*0.9), "exp": f"Mechanical overload detected ({data.Current}A > 8A nominal baseline). High impeller friction due to sand drag.", "sop": "Extract pump from borehole. Clean intake strainers and inspect impellers for abrasive wear."},
        3: {"status": "Critical", "issue": "Severe Cavitation", "health": int(100 - conf*0.7), "exp": f"Extreme vibrations ({data.Vibration}mm/s) exceeding ISO 10816 limits. Net Positive Suction Head (NPSH) critically low.", "sop": "Reduce MPPT controller frequency to stabilize fluid velocity and check pump submergence depth."}
    }[pred]

    # XAI (Explainable AI) Calculation
    ref = np.array([800, 8.0, 35, 1.5, 50, 30])
    weights = np.abs(raw[0] - ref) / (ref + 0.1)
    xai_dict = dict(zip(feature_names, (weights / np.sum(weights)).tolist()))
    
    return {**diag, "xai": xai_dict, "confidence": f"{conf:.1f}%"}

# NEW ENDPOINT: Serving E4C Report Knowledge to the Chatbot
@app.get("/e4c_knowledge")
def get_e4c_knowledge():
    return {
        "sds_t_128": "SDS-T-128 is a low-cost positive displacement solar submersible pump operating on 12-30 V DC, suitable for 4-inch casings.",
        "shurflo_9300": "Shurflo 9300 Series is a solar submersible pump for low-flow systems. Performance: 82 gal/h at 230 ft using 24 VDC and a min 155 W solar array.",
        "gf1": "GF1 Submersible Solar Pump operates at 12/24 V DC for remote irrigation farming.",
        "rainmaker_2": "RainMaker 2 pumps from 70m depth at 50 L/min, with a maximum motor current of 8 A and a voltage range of 24-34 V.",
        "pumpmakers": "Pumpmakers DIY Solar Pump is an open-source above-ground pump using a submersible piston and surface crankshaft. Operates up to 100m depth, 18,000 L/day."
    }