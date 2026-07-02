# 🌏 India Climate Digital Twin
### AI-Powered Digital Twin of India's Climate · NRSC / ISRO Hackathon — Problem Statement 5

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Overview
Real-time interactive Digital Twin of India's climate using GraphCast GNN + U-Net CNN bias correction, validated against IMD RF25 observations.

## 🔗 Live Demo
**[View the app on Streamlit Cloud](https://your-app.streamlit.app)**
*(link will be updated after deployment)*

## Pipeline
ERA5 → GraphCast GNN → India Domain (0.25°) → U-Net CNN Bias Correction → IMD RF25 Validation

## Features
- 🌐 3D Globe view of India
- 🗺️ 2D Map visualization  
- ⛰️ 3D Surface plots
- 📊 Rainfall & Temperature Monitor
- 🌧️ Flood Event Replay (Kerala 2018, Uttarakhand 2013)
- 🔮 What-If Climate Scenario Simulator

## Run Locally
```bash
pip install -r requirements.txt
python3 -m streamlit run climate_twin_v2.py
```

## Data Sources
- ERA5 Reanalysis (ECMWF)
- GraphCast GNN (DeepMind)
- IMD RF25 (0.25° gridded)
- Survey of India boundary shapefile

## Built for
NRSC / ISRO Hackathon · Symbiosis Institute of Technology, Hyderabad
