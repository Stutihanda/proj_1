# 🛡️ Climate Guardian AI

**An AI-powered environmental risk intelligence platform that combines live weather telemetry, multi-dataset AutoML pipelines, and conversational AI to predict and visualize regional outbreak/hazard risk.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![n8n](https://img.shields.io/badge/n8n-Workflow%20Automation-EA4B71?logo=n8n&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Capstone%20Project-yellow)

---

## 📖 Overview

**Climate Guardian AI** is a full-stack, agent-driven analytics platform built as a capstone project. It ingests environmental/climate telemetry data, runs an automated machine learning pipeline to classify regional risk levels, and surfaces the results through an interactive command-center dashboard — complete with correlation analysis, SHAP-based explainability, live weather lookups, and a conversational AI assistant powered by **n8n + Ollama**.

The system is designed around the idea that climate variables (temperature, humidity, precipitation) correlate with region-level risk (e.g., disease vector activity, environmental hazard exposure), and it automates the full loop: **ingest → model → visualize → explain → chat.**

---

## ✨ Key Features

- 📁 **Multi-Dataset Ingestion** — Upload and merge multiple CSV telemetry files on the fly, with automatic schema reconciliation.
- 🤖 **AutoML Pipeline Orchestrator** — Automatically compares multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) and selects the best performer.
- 📊 **Outbreak Command Center** — Executive dashboard with live KPIs, risk-tier filtering, and exportable summaries.
- 📈 **Advanced Analytics Studio** — Correlation heatmaps, bivariate driver analysis, distribution plots, 3D cluster visualization, and SHAP-based feature impact charts.
- 📡 **Live Weather Radar** — Real-time weather lookups via the Open-Meteo API (temperature, humidity, precipitation, wind speed).
- 🤖 **AI Risk Assistant** — Natural-language Q&A about risk factors and model insights, powered by an **n8n workflow calling a local Ollama LLM**, with an offline fallback mode.
- 📝 **Execution Logs & Model Registry** — Full audit trail of pipeline runs, users, and model performance.
- 🔐 **User Authentication** — Signup/login with bcrypt-hashed credentials via a lightweight SQLite store.

---

## 🏗️ Architecture

```
┌────────────────────┐        ┌──────────────────────┐        ┌───────────────────┐
│  Streamlit Frontend │ <----> │   FastAPI Backend     │ <----> │   SQLite Database  │
│  (streamlit_app.py) │  HTTP  │ (main.py / api.py)    │        │  (users, logs)     │
└────────┬────────────┘        └──────────┬────────────┘        └───────────────────┘
         │                                 │
         │                                 ├──> Open-Meteo API (live weather + geocoding)
         │                                 │
         │                                 └──> n8n Webhook ──> Ollama LLM (AI Risk Assistant)
         │
         └──> Multi-Agent Pipeline (agents/) ──> AutoML model comparison + SHAP explainability
```

**Flow:**
1. User uploads one or more CSV telemetry datasets via the Streamlit UI.
2. The FastAPI backend merges the datasets and runs them through the pipeline orchestrator.
3. The orchestrator trains/evaluates candidate models, selects the best one, and computes top feature importances.
4. Results are logged to SQLite and rendered back into the dashboard (KPIs, charts, SHAP plots).
5. The AI Risk Assistant tab routes natural-language questions to an n8n workflow, which queries a local Ollama model and returns a contextual answer.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend / Dashboard | Streamlit, Plotly Express |
| Backend / API | FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy |
| Authentication | Passlib (bcrypt) |
| Database | SQLite |
| Workflow Automation | n8n |
| Conversational AI | Ollama (local LLM) |
| External Data | Open-Meteo Weather & Geocoding API |

---

## 📂 Project Structure

```
proj_1/
├── agents/              # Multi-agent pipeline logic (model orchestration, SHAP, risk scoring)
├── datasets/            # Sample / uploaded CSV telemetry datasets
├── api.py               # Core FastAPI service (pipeline execution, weather, AI Q&A)
├── main.py               # FastAPI backend (auth, pipeline orchestrator, weather, AI chat)
├── streamlit_app.py     # Streamlit dashboard (all UI tabs and visualizations)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- (Optional, for the AI Assistant) [n8n](https://n8n.io/) running locally + [Ollama](https://ollama.com/) with a pulled model

### 1. Clone the repository

```bash
git clone https://github.com/Stutihanda/proj_1.git
cd proj_1
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn streamlit pandas numpy plotly passlib requests
```

> 💡 Tip: Consider adding a `requirements.txt` to freeze these versions for reproducibility.

### 3. Run the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

### 4. Run the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`.

### 5. (Optional) Enable the AI Risk Assistant

- Start Ollama locally and pull a model (e.g., `ollama pull llama3`).
- Import/run an n8n workflow exposing a webhook at `http://localhost:5678/webhook/run-pipeline`.
- Alternatively, enable **"Use Local AI Fallback"** in the sidebar to skip the n8n/Ollama dependency entirely.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/signup` | Register a new user |
| `POST` | `/login` | Authenticate a user |
| `POST` | `/run-pipeline` | Upload CSV(s) and trigger the AutoML pipeline |
| `GET` | `/weather/{city}` | Fetch live weather data for a city |
| `GET` | `/history/{username}` | Retrieve a user's execution history |
| `POST` | `/ask` | Ask the AI assistant a risk-related question |

---

## 📊 Dashboard Tabs

1. **Outbreak Command Center** — High-level executive telemetry and risk-tier monitoring.
2. **Live Weather Radar** — Real-time meteorological feeds.
3. **Multi-Dataset Ingestion** — Upload and combine CSVs, trigger the pipeline.
4. **Execution Logs** — Audit trail of runs and model performance.
5. **Advanced Analytics Studio** — Correlation matrices, bivariate plots, 3D clustering, SHAP impact.
6. **AI Risk Assistant** — Conversational Q&A on risk factors and model behavior.

---

## 🗺️ Roadmap

- [ ] Add a `requirements.txt` / `pyproject.toml` for dependency management
- [ ] Containerize with Docker Compose (backend + Streamlit + n8n + Ollama)
- [ ] Replace simulated model scores with a fully persisted model registry
- [ ] Add automated tests for the API layer
- [ ] Deploy a hosted demo (Streamlit Community Cloud / Render)

---

## 🎓 About This Project

This project was built as a **capstone project**, demonstrating an end-to-end applied ML system: data ingestion, automated model selection, explainability (SHAP), live external API integration, and an LLM-powered assistant — all wrapped in a production-style dashboard.

---

## 👩‍💻 Author

**Stuti Handa**
GitHub: [@Stutihanda](https://github.com/Stutihanda)

---

## 📄 License

This project is licensed under the MIT License — feel free to use and adapt it for your own learning.
