# Retail Demand Forecasting - AI Intelligence System

**Sanjay Sarella** | M.S. Data Analytics, Oklahoma City University

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-orange)](https://facebook.github.io/prophet)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange)](https://xgboost.readthedocs.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3--70B-green)](https://console.groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://streamlit.io)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-blue?logo=googlecloud)](https://cloud.google.com)
[![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)](https://powerbi.microsoft.com)

## Live App
**[Launch App](https://retail-demand-app-646m5mi6fq-uc.a.run.app/)**

---

## The Problem

A 45-store regional retailer loses ~$4.2M annually to demand forecast errors. Stockouts during Thanksgiving and Christmas weeks leave revenue on the table. Overstock in slow periods ties up 25–30% of inventory value in carrying costs. Store managers have no data-driven tool to act on - decisions are made on intuition.

This system changes that.

---

## What This Does

An end-to-end retail demand intelligence system built on real Walmart weekly sales data - 45 stores, 2.5 years, 421,570 records. It forecasts weekly demand 4 weeks ahead, explains every forecast using SHAP, flags stockout risk departments, drafts purchase orders automatically, and generates plain-English inventory briefs via a 5-node LangGraph multi-agent pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — ML FORECASTING                                   │
│  Prophet → seasonality + holiday effects                    │
│  XGBoost → CPI, markdowns, store type, lag features         │
│  Weighted ensemble (40% Prophet + 60% XGBoost)              │
│  SHAP TreeExplainer → per-store feature attribution         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  LAYER 2 — RAG KNOWLEDGE BASE                               │
│  ChromaDB vector store                                      │
│  SHAP summaries + retail strategy documents embedded        │
│  Semantic retrieval on every agent query                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  LAYER 3 — LANGGRAPH 5-NODE MULTI-AGENT PIPELINE            │
│                                                             │
│  Node 1: Forecast Agent → ForecastOutput (Pydantic)         │
│  Node 2: SHAP Explainer Agent → SHAPExplanation             │
│  Node 3: Inventory Optimizer → tool-calling                 │
│          calculate_reorder_qty()                            │
│          flag_stockout_risk()                               │
│          draft_purchase_order()                             │
│  Node 4: Critic Agent → reflection loop (max 2 retries)     │
│  Node 5: Executive Strategist → Groq + Llama 3.3-70B        │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Dataset | 45 Walmart stores · 2.5 years · 421,570 records |
| Forecast horizon | 4 weeks ahead per store × department |
| Holiday sales lift detected | +7.2% (SHAP verified) |
| High-risk departments flagged (Store 1) | 5 departments |
| Recommended reorder quantity | 31,503 units |
| Critic Agent quality score | 100% pass rate |
| Estimated prevented stockout losses | $1.3M annually |
| Deployment | GCP Cloud Run |
| Total cost | $0 — fully open source |

---

## What Makes This Different

**Prophet + XGBoost ensemble — right tool for each job**
Prophet handles time-series seasonality and holiday effects cleanly. XGBoost handles tabular features that Prophet ignores - CPI, unemployment, markdowns, store type. Neither model alone is as strong as both together. The ensemble architecture was a deliberate design choice, not a default.

**Tool-calling agents with real decision authority**
The Inventory Optimizer agent does not describe what should happen - it calls actual Python functions and produces a structured purchase order with line items, quantities, and estimated values. The output is actionable, not advisory.

**Reflection loop with a Critic Agent**
Before the Executive Strategist generates a brief, a dedicated Critic Agent scores the output against a 4-point quality rubric. If confidence is below 75%, the output is rejected and sent back for revision automatically. This mirrors how production AI quality gates work.

**Pydantic-typed outputs at every node**
Every agent node returns a typed schema - ForecastOutput, SHAPExplanation, InventoryDecision, CriticScore, StrategyBrief. No raw text passed between nodes. The pipeline is robust and auditable end to end.

**SQL demonstrated explicitly**
The EDA notebook loads data into SQLite and runs 4 business queries - holiday premium by store type, top departments by sales, highest-variance stores flagged as stockout risks. SQL is shown in the work, not just listed as a skill.

---

## How to Use the App

1. Open the **[live app](https://retail-demand-app-646m5mi6fq-uc.a.run.app/)**
2. Select a store (1–45) and department from the sidebar
3. Click **Analyse this store** to run the full 5-node pipeline
4. Navigate across 5 tabs:
   - **Forecast vs Actual** — XGBoost predictions vs real sales with MAE and MAPE
   - **Store Risk Heatmap** — 45-store scatter showing HIGH / MEDIUM / LOW stockout risk
   - **Holiday Impact** — Sales lift by holiday event with actual vs forecast comparison
   - **SHAP Feature Importance** — Top 10 demand drivers by mean absolute SHAP value
   - **AI Strategy Brief** — Live pipeline output with executive summary, findings, and purchase order draft
5. Use the **AI Chat** page for plain-English questions about any store

---

## Tech Stack

| Category | Tools |
|---|---|
| ML & Forecasting | Prophet · XGBoost · Scikit-learn · SHAP |
| Agentic AI | LangGraph · LangChain · ChromaDB · Pydantic · Groq + Llama 3.3-70B |
| App | Streamlit |
| Visualization | Power BI · Plotly |
| Deployment | Docker · GCP Cloud Run |
| Data | SQLite · Pandas · NumPy · Kaggle Walmart Dataset |
| Dev | Python 3.11 · Jupyter · Git |

**100% open source. Zero cost.**

---

## Project Structure

```
retail-demand-forecasting/
├── agents/
│   ├── pipeline.py          ← LangGraph StateGraph (5 nodes + reflection loop)
│   ├── nodes.py             ← All 5 agent node definitions + Pydantic schemas
│   ├── tools.py             ← Tool-calling functions (reorder qty, risk, PO draft)
│   └── chroma_ingest.py     ← ChromaDB ingestion
├── app/
│   ├── Demand_Intelligence.py   ← Main Streamlit dashboard
│   └── pages/
│       └── 01_AI_Chat.py        ← Dedicated AI chat agent
├── data/
│   ├── raw/                 ← Kaggle CSVs (not tracked — see dataset note)
│   └── processed/           ← Models, predictions, SHAP artifacts
├── docs/                    ← SHAP plots + Power BI dashboard
├── notebooks/
│   ├── 01_eda.ipynb         ← EDA + SQL queries + data cleaning
│   ├── 02_models.ipynb      ← Prophet + XGBoost + ensemble evaluation
│   └── 03_shap.ipynb        ← SHAP analysis + ChromaDB ingestion
├── tableau/exports/         ← CSVs for Power BI
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-Walmart%20Sales%20Forecast-blue?logo=kaggle)](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

Free Kaggle account required. Download and place in `data/raw/`:
```
train.csv · features.csv · stores.csv
```

---

## Power BI Dashboard
5-page dashboard covering forecast vs actual, store risk heatmap, holiday impact analysis, and SHAP feature importance. File available in `docs/Retail_Demand_Intelligence.pbix`.

---

## Author

**Sanjay Sarella**
M.S. Data Analytics — Oklahoma City University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sanjay%20Sarella-blue?logo=linkedin)](https://linkedin.com/in/sanjaysarella)
[![GitHub](https://img.shields.io/badge/GitHub-SanjaySarella-black?logo=github)](https://github.com/SanjaySarella)
