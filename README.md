# 🌏 India Climate Digital Twin
### AI-Powered Digital Twin of India's Climate · NRSC / ISRO Hackathon — Problem Statement 5

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://india-climate-digital-twin-bpy9vcdf4cfvzdxqc6pfz4.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 🔗 Live Demo

**[👉 Open the App on Streamlit Cloud](https://india-climate-digital-twin-bpy9vcdf4cfvzdxqc6pfz4.streamlit.app/)**

---

## 📌 About the Project

India is one of the most climate-vulnerable countries in the world — from catastrophic monsoon floods in Kerala and Uttarakhand to droughts across Marathwada. This project builds an **AI-Powered Digital Twin of India's Climate** — a live, interactive simulation platform combining deep learning with gridded observational data to replicate, monitor, and predict India's climate at 0.25° resolution.

Developed for the **NRSC / ISRO Hackathon (Problem Statement 5)**.

---

## 🧠 Our Approach

We use a **two-stage deep learning pipeline**:

**Stage 1 — GraphCast GNN:** DeepMind's Graph Neural Network trained on 40 years of ERA5 data. Produces global forecasts in under 60 seconds. We run it in frozen inference mode and extract the India domain.

**Stage 2 — U-Net CNN (Bias Correction):** GraphCast has systematic biases over India. A U-Net CNN with 11 input channels learns and corrects these biases, validated against IMD RF25 at 0.25° resolution.

### Why This is Better

| Approach | Speed | Resolution | Bias Fix |
|----------|-------|-----------|---------|
| Traditional NWP (WRF) | Days | 0.1–0.5° | Manual |
| Plain GraphCast | Seconds | 1° | None |
| **Ours: GraphCast + U-Net** | **Minutes** | **0.25°** | **AI-learned** |

---

## 🔄 Pipeline

ERA5 Reanalysis → GraphCast GNN → India Domain Extraction → U-Net CNN Bias Correction → IMD RF25 Validation → Streamlit Dashboard

---

## 🖥️ Features

| Mode | Description |
|------|-------------|
| 🌐 3D Globe | Orthographic globe centred on India with live data |
| 📊 Rainfall Monitor | GraphCast vs CNN vs IMD with MAE and bias metrics |
| 🌡️ Temperature Monitor | Monthly fields with bias correction stats |
| 🌧️ Flood Event Replay | Kerala 2018 · Uttarakhand 2013 |
| 🔮 What-If Scenario | Simulate rainfall/temperature shifts with risk alerts |

Three views in every mode: 🌍 Globe · 🗺️ 2D Map · ⛰️ 3D Surface

---

## ⚙️ Run Locally

```bash
git clone https://github.com/vinaychakravarthyaddanki1704/India-climate-digital-twin.git
cd India-climate-digital-twin
pip install -r requirements.txt
python3 -m streamlit run climate_twin_v2.py
```

---

## 📦 Tech Stack

GraphCast GNN · U-Net CNN · Streamlit · Plotly · GeoPandas · Shapely · NumPy · Matplotlib

---

## 📡 Data Sources

- ERA5 Reanalysis — ECMWF / Copernicus (0.25°, Hourly)
- IMD RF25 — India Meteorological Department (0.25°, Daily)
- Survey of India — State boundary shapefile

---

## 🏆 Hackathon

NRSC / ISRO National Hackathon · PS-5 · Symbiosis Institute of Technology, Hyderabad

---

## 📄 License

MIT License
