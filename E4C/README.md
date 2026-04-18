# THE STRIVERS - Solar Pump Intelligent Monitoring

## Description
AI-powered diagnostic system for community solar pumps (E4C Track 1). 
Detects Dry Running and Sand Clogging using Random Forest classification and XAI.

## Installation
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the AI Engine: `uvicorn api:app --reload`
3. Launch the Dashboard: `streamlit run app.py`

## Features
- Real-time telemetry visualization.
- Explainable AI (XAI) for engineering insights.
- Automated SOP (Standard Operating Procedures) for maintenance.