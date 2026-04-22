# Retail Demand Forecasting — AI Intelligence System

> **Built by Sanjay Sarella** | M.S. Data Analytics, Oklahoma City University

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3--70B-green)](https://console.groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-gray)](LICENSE)

---

## The Business Problem

A 45-store regional retailer loses **~$4.2M annually** to demand forecast errors.

- Stockouts during Thanksgiving and Christmas weeks leave revenue on the table
- Overstock in slow periods ties up 25–30% of inventory value in carrying costs
- Store managers have no data-driven tool to act on — they rely on intuition

**This system changes that.**

---

## What This System Does

This is not a basic forecasting model. It is a three-layer AI system that:

1. **Forecasts** weekly demand per store and department 4 weeks ahead
2. **Explains** every forecast in plain English using SHAP — "holiday weeks add $3,200 to Store 1's weekly sales"
3. **Acts** — the AI agent drafts purchase orders, flags stockout risk, and generates executive briefs automatically

**Modelled business impact: $1.3M in prevented stockout losses annually across a 45-store chain.**

---

## Architecture — Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — ML FORECASTING                                   │
│                                                             │
│  Prophet  ──────────────────────────────────────────────►  │
│  (seasonality + holiday effects)          Weighted Ensemble │
│                                           (40% + 60%)       │
│  XGBoost ───────────────────────────────────────────────►  │
│  (CPI, markdowns, store type, lag features)                 │
│                                                             │
│  SHAP TreeExplainer → feature importance per store          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  LAYER 2 — RAG KNOWLEDGE BASE                               │
│                                                             │
│  ChromaDB vector store                                      │
│  SHAP summaries + retail strategy documents embedded        │
│  Semantic retrieval on every agent query                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  LAYER 3 — LANGGRAPH 5-NODE MULTI-AGENT PIPELINE            │
│                                                             │
│  Node 1: Forecast Agent                                     │
│    └─► Runs ensemble, outputs ForecastOutput (Pydantic)     │
│                                                             │
│  Node 2: SHAP Explainer Agent                               │
│    └─► Surfaces top drivers, outputs SHAPExplanation        │
│                                                             │
│  Node 3: Inventory Optimizer Agent  ← TOOL-CALLING          │
│    └─► Calls calculate_reorder_qty()                        │
│    └─► Calls flag_stockout_risk()                           │
│    └─► Calls draft_purchase_order()                         │
│    └─► Outputs InventoryDecision (Pydantic)                 │
│                                                             │
│  Node 4: Critic Agent  ← REFLECTION LOOP                   │
│    └─► Scores output against 4-point rubric                 │
│    └─► Routes back to Node 3 if score < 0.75               │
│    └─► Max 2 retry cycles                                   │
│                                                             │
│  Node 5: Executive Strategist Agent                         │
│    └─► RAG + Groq + Llama 3.3-70B                          │
│    └─► Generates plain-English strategy brief               │
│    └─► Answers store manager queries via Streamlit chat     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Dataset | 45 Walmart stores · 2.5 years · 421,570 rows |
| Forecast horizon | 4 weeks ahead per store × department |
| Holiday sales lift detected | +7.2% average (SHAP verified) |
| High-risk departments flagged (Store 1) | 5 departments |
| Recommended reorder quantity (Store 1) | 31,503 units |
| Critic Agent quality score | 100% pass rate |
| Estimated prevented stockout losses | $1.3M annually |
| Pipeline cost | $0 — fully open source |

---

## What Makes This Different


**I gave the agents real decision-making power**
The Inventory Optimizer agent does not just describe what a store manager should do.
It calls actual Python functions, calculates reorder quantities from real sales data,
and produces a formatted purchase order with line items and estimated values.
The output is actionable, not advisory.

**I built a self-correcting pipeline**
Before the final strategy brief is generated, a dedicated Critic Agent reviews the
inventory decision against a 4-point quality rubric. If the output scores below 75%,
it is automatically sent back for revision. I did this because real AI systems in
production need quality gates — not just outputs.

**I enforced structure at every step**
Every agent in the pipeline returns a typed Pydantic schema instead of raw text.
This means no ambiguous outputs, no silent failures, and a pipeline that can be
audited node by node. This is how I think about engineering discipline in AI systems.

**I made SQL visible, not hidden**
The EDA notebook does not just use Pandas. I loaded the data into SQLite and wrote
4 business queries — holiday sales premium by store type, top-performing departments,
and highest-variance stores flagged as stockout risks. SQL is a core skill and I
wanted it demonstrated explicitly in the work, not just listed on a resume.

---

## Tech Stack

| Category | Tools |
|---|---|
| ML & Forecasting | Prophet · XGBoost · Scikit-learn · SHAP |
| Agentic AI | LangGraph · LangChain · ChromaDB · Pydantic |
| Language Model | Groq API · Llama 3.3-70B (free tier) |
| Evaluation | DeepEval |
| App | Streamlit |
| Visualization | Power BI · Plotly · Matplotlib · Seaborn |
| Data | SQLite · Pandas · NumPy |
| Dev | Python 3.11 · Jupyter · Git |

**100% open source. Zero cost.**

---

## Streamlit App — 5 Tabs + AI Chat

| Tab | What it shows |
|---|---|
| Forecast vs actual | XGBoost predictions vs real sales, MAE, MAPE per store-dept |
| Store risk heatmap | 45-store scatter — HIGH / MEDIUM / LOW stockout risk |
| Holiday impact | Sales lift by holiday event (Super Bowl, Thanksgiving, Christmas) |
| SHAP feature importance | Top 10 drivers by mean absolute SHAP value |
| AI strategy brief | Live pipeline output — executive summary, findings, purchase order |
| AI Chat page | Dedicated chatbot — answers specific questions about any store |

---

## Project Structure

```
retail-demand-forecasting/
├── agents/
│   ├── pipeline.py          ← LangGraph StateGraph (5 nodes + reflection loop)
│   ├── nodes.py             ← All 5 agent node definitions + Pydantic schemas
│   ├── tools.py             ← Tool-calling functions (reorder qty, risk, PO draft)
│   └── chroma_ingest.py     ← ChromaDB ingestion (SHAP + strategy docs)
├── app/
│   ├── Demand_Intelligence.py   ← Main Streamlit dashboard
│   └── pages/
│       └── 01_AI_Chat.py        ← Dedicated AI chat agent
├── data/
│   ├── raw/                 ← Kaggle CSVs (see setup — not tracked in git)
│   └── processed/           ← Generated artifacts (models, predictions, SHAP)
├── docs/                    ← SHAP plots (beeswarm, waterfall, bar, holiday)
├── notebooks/
│   ├── 01_eda.ipynb         ← EDA + SQL queries + data cleaning
│   ├── 02_models.ipynb      ← Prophet + XGBoost + ensemble + evaluation
│   └── 03_shap.ipynb        ← SHAP analysis + ChromaDB ingest + exports
├── tableau/exports/         ← CSVs for Power BI dashboard
├── .env                     ← API keys (not tracked)
└── requirements.txt
```

---

## Setup — Run It Yourself

### 1. Clone the repo
```bash
git clone https://github.com/SanjaySarella/retail-demand-forecasting.git
cd retail-demand-forecasting
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Free key at: https://console.groq.com — takes 2 minutes, no card required.

### 5. Download the Walmart dataset

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Walmart%20Sales%20Forecast-blue?logo=kaggle)](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

Free Kaggle account required. Place these three files in `data/raw/`:
```
data/raw/train.csv
data/raw/features.csv
data/raw/stores.csv
```

### 6. Run notebooks in order
```bash
jupyter notebook
```
1. `notebooks/01_eda.ipynb` — EDA, SQL layer, data cleaning
2. `notebooks/02_models.ipynb` — Prophet + XGBoost + ensemble
3. `notebooks/03_shap.ipynb` — SHAP + ChromaDB ingestion

### 7. Run the LangGraph pipeline
```bash
python agents/pipeline.py
```

### 8. Launch the app
```bash
streamlit run app/Demand_Intelligence.py
```
Open **http://localhost:8501**

---


## Author

**Sanjay Sarella**
M.S. Data Analytics — Oklahoma City University
Positioning: Business Strategy + Data Analytics + Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sanjay%20Sarella-blue?logo=linkedin)](https://linkedin.com/in/sanjaysarella)
[![GitHub](https://img.shields.io/badge/GitHub-SanjaySarella-black?logo=github)](https://github.com/SanjaySarella)
