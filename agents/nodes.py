import json
import os
import pandas as pd
import numpy as np
import pickle
from typing import Optional
from pydantic import BaseModel, Field
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.tools import calculate_reorder_qty, flag_stockout_risk, draft_purchase_order
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent
load_dotenv(BASE_PATH / '.env')

CHROMA_PATH = str(BASE_PATH / 'data' / 'processed' / 'chromadb')
FEATURES = [
    'Store','Dept','Week','Month','Quarter','Year',
    'IsHoliday','Type_enc','Size',
    'Temperature','Fuel_Price','CPI','Unemployment',
    'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','MarkDown_Total',
    'lag_1','lag_4','lag_8','lag_52',
    'rolling_mean_4','rolling_mean_12','rolling_std_4'
]


# ── Pydantic output schemas ──────────────────────────────────────────────────

class ForecastOutput(BaseModel):
    store_id        : int
    dept            : int
    forecast_4wk    : list[float]
    avg_forecast    : float
    trend           : str
    confidence      : float

class SHAPExplanation(BaseModel):
    store_id        : int
    dept            : int
    top_driver      : str
    top_driver_value: float
    holiday_impact  : float
    lag_impact      : float
    summary_text    : str

class InventoryDecision(BaseModel):
    store_id        : int
    reorder_qty     : float
    risk_level      : str
    risk_depts      : list[int]
    purchase_order  : dict
    confidence      : float
    justification   : str

class CriticScore(BaseModel):
    passed          : bool
    score           : float
    feedback        : str
    retry_count     : int

class StrategyBrief(BaseModel):
    store_id        : int
    executive_summary: str
    key_findings    : list[str]
    recommended_actions: list[str]
    dollar_impact   : str


# ── Helper: build features for a single store-dept ───────────────────────────

def build_store_features(df, store_id, dept):
    df = df.copy().sort_values(['Store','Dept','Date'])

    df['Week']        = df['Date'].dt.isocalendar().week.astype(int)
    df['Month']       = df['Date'].dt.month
    df['Quarter']     = df['Date'].dt.quarter
    df['Year']        = df['Date'].dt.year
    df['IsHoliday']   = df['IsHoliday'].astype(int)

    grp = df.groupby(['Store','Dept'])['Weekly_Sales']
    df['lag_1']       = grp.shift(1)
    df['lag_4']       = grp.shift(4)
    df['lag_8']       = grp.shift(8)
    df['lag_52']      = grp.shift(52)
    df['rolling_mean_4']  = grp.shift(1).transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['rolling_mean_12'] = grp.shift(1).transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_4']   = grp.shift(1).transform(lambda x: x.rolling(4,  min_periods=1).std().fillna(0))
    df['MarkDown_Total']  = df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1)

    le = LabelEncoder()
    df['Type_enc'] = le.fit_transform(df['Type'])

    store_dept = df[(df['Store'] == store_id) & (df['Dept'] == dept)].dropna(subset=['lag_1'])
    return store_dept


# ── Node 1: Forecast Agent ────────────────────────────────────────────────────

def forecast_agent(state: dict) -> dict:
    print(f"\n[Node 1] Forecast Agent — Store {state['store_id']}, Dept {state['dept']}")

    csv_path = BASE_PATH / 'data' / 'processed' / 'walmart_clean.csv'
    df       = pd.read_csv(csv_path, parse_dates=['Date'])
    xgb      = XGBRegressor()
    model_path = BASE_PATH / 'data' / 'processed' / 'xgb_model.json'
    xgb.load_model(str(model_path))

    store_df = build_store_features(df, state['store_id'], state['dept'])

    if store_df.empty:
        state['error'] = f"No data for Store {state['store_id']}, Dept {state['dept']}"
        return state

    last_rows   = store_df.tail(4)
    X_last      = last_rows[FEATURES]
    predictions = xgb.predict(X_last).tolist()
    avg_pred    = float(np.mean(predictions))

    recent_avg  = float(store_df.tail(8)['Weekly_Sales'].mean())
    trend       = 'upward' if avg_pred > recent_avg * 1.05 else \
                  'downward' if avg_pred < recent_avg * 0.95 else 'stable'

    output = ForecastOutput(
        store_id     = state['store_id'],
        dept         = state['dept'],
        forecast_4wk = [round(p, 2) for p in predictions],
        avg_forecast = round(avg_pred, 2),
        trend        = trend,
        confidence   = 0.85
    )

    print(f"  Forecast: {[round(p,0) for p in predictions]}")
    print(f"  Trend: {trend} | Avg: ${avg_pred:,.0f}")

    state['forecast'] = output.dict()
    return state


# ── Node 2: SHAP Explainer Agent ──────────────────────────────────────────────

def shap_explainer_agent(state: dict) -> dict:
    print(f"\n[Node 2] SHAP Explainer Agent — Store {state['store_id']}")

    shap_path = BASE_PATH / 'data' / 'processed' / 'shap_summary.json'
    with open(shap_path, 'r') as f:
        shap_data = json.load(f)

    store_summary = next(
        (s for s in shap_data.get('store_summaries', []) if s['store_id'] == state['store_id']),
        None
    )

    if store_summary:
        top_driver       = store_summary['top_3_drivers'][0][0]
        top_driver_value = store_summary['top_3_drivers'][0][1]
        holiday_impact   = store_summary['holiday_impact']
        lag_impact       = store_summary['lag1_impact']
    else:
        top_driver       = shap_data['top_feature']
        top_driver_value = shap_data['global_top10_features'][0]['mean_abs_shap']
        holiday_impact   = shap_data['holiday_avg_shap']
        lag_impact       = 0.0

    summary_text = (
        f"For Store {state['store_id']}, the top demand driver is '{top_driver}' "
        f"with an average SHAP impact of ${top_driver_value:,.0f}. "
        f"Holiday weeks add ${holiday_impact:,.0f} on average to weekly sales. "
        f"Recent sales history (lag_1) contributes ${lag_impact:,.0f}."
    )

    output = SHAPExplanation(
        store_id         = state['store_id'],
        dept             = state['dept'],
        top_driver       = top_driver,
        top_driver_value = float(top_driver_value),
        holiday_impact   = float(holiday_impact),
        lag_impact       = float(lag_impact),
        summary_text     = summary_text
    )

    print(f"  Top driver: {top_driver} (${top_driver_value:,.0f})")
    print(f"  Holiday impact: ${holiday_impact:,.0f}")

    state['shap_explanation'] = output.dict()
    return state


# ── Node 3: Inventory Optimizer Agent (tool-calling) ─────────────────────────

def inventory_optimizer_agent(state: dict) -> dict:
    retry = state.get('retry_count', 0)
    print(f"\n[Node 3] Inventory Optimizer Agent — Store {state['store_id']} (attempt {retry+1})")

    store_id = state['store_id']
    dept     = state['dept']

    # Call tool 1 — reorder quantity
    reorder_result = json.loads(calculate_reorder_qty.invoke({
        'store_id': store_id,
        'dept'    : dept
    }))

    # Call tool 2 — stockout risk flags
    risk_result = json.loads(flag_stockout_risk.invoke({
        'store_id' : store_id,
        'threshold': 0.3
    }))

    # Call tool 3 — draft purchase order
    po_result = json.loads(draft_purchase_order.invoke({
        'store_id': store_id
    }))

    reorder_qty  = reorder_result.get('recommended_reorder_qty', 0)
    risk_level   = 'HIGH' if risk_result.get('high_risk_count', 0) > 2 else \
                   'MEDIUM' if risk_result.get('medium_risk_count', 0) > 0 else 'LOW'
    risk_depts   = [d['dept'] for d in risk_result.get('high_risk_depts', [])]
    confidence   = reorder_result.get('confidence', 0.8)

    justification = (
        f"Reorder quantity of {reorder_qty:,.0f} units calculated from "
        f"{reorder_result.get('based_on_weeks', 8)} weeks of sales history. "
        f"Safety stock set at 1.5 standard deviations above mean sales of "
        f"${reorder_result.get('avg_weekly_sales', 0):,.0f}. "
        f"Risk assessment: {risk_result.get('high_risk_count', 0)} high-risk departments, "
        f"{risk_result.get('medium_risk_count', 0)} medium-risk departments identified."
    )

    output = InventoryDecision(
        store_id      = store_id,
        reorder_qty   = reorder_qty,
        risk_level    = risk_level,
        risk_depts    = risk_depts,
        purchase_order= po_result,
        confidence    = confidence,
        justification = justification
    )

    print(f"  Reorder qty: {reorder_qty:,.0f} | Risk: {risk_level}")
    print(f"  High-risk depts: {risk_depts}")

    state['inventory_decision'] = output.dict()
    state['retry_count']        = retry
    return state


# ── Node 4: Critic Agent (reflection loop) ───────────────────────────────────

def critic_agent(state: dict) -> dict:
    print(f"\n[Node 4] Critic Agent — reviewing inventory decision")

    decision = state.get('inventory_decision', {})
    score    = 0.0
    feedback = []

    # Check 1 — reorder quantity is quantified
    if decision.get('reorder_qty', 0) > 0:
        score += 0.25
    else:
        feedback.append("Reorder quantity is zero or missing — not actionable.")

    # Check 2 — risk level is defined
    if decision.get('risk_level') in ['HIGH','MEDIUM','LOW']:
        score += 0.25
    else:
        feedback.append("Risk level not properly defined.")

    # Check 3 — justification is substantive
    justification = decision.get('justification', '')
    if len(justification) > 80 and '$' in justification:
        score += 0.25
    else:
        feedback.append("Justification lacks dollar-quantified evidence.")

    # Check 4 — purchase order is present and has line items
    po = decision.get('purchase_order', {})
    if po.get('line_items') and len(po['line_items']) > 0:
        score += 0.25
    else:
        feedback.append("Purchase order is missing or has no line items.")

    passed        = score >= 0.75
    retry_count   = state.get('retry_count', 0)
    feedback_text = ' '.join(feedback) if feedback else "All checks passed. Output is actionable."

    critic_output = CriticScore(
        passed      = passed,
        score       = round(score, 2),
        feedback    = feedback_text,
        retry_count = retry_count
    )

    print(f"  Score: {score:.2f} | Passed: {passed}")
    if feedback:
        print(f"  Feedback: {feedback_text}")

    state['critic_score'] = critic_output.dict()
    return state


def critic_router(state: dict) -> str:
    """Routes to executive strategist if passed, back to optimizer if failed."""
    critic  = state.get('critic_score', {})
    retries = state.get('retry_count', 0)

    if critic.get('passed') or retries >= 2:
        print("  → Routing to Executive Strategist")
        return 'executive_strategist'
    else:
        print(f"  → Routing back to Inventory Optimizer (retry {retries + 1})")
        state['retry_count'] = retries + 1
        return 'inventory_optimizer'


# ── Node 5: Executive Strategist Agent ───────────────────────────────────────

def executive_strategist_agent(state: dict) -> dict:
    print(f"\n[Node 5] Executive Strategist Agent — generating brief")

    # Pull context from ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = embedding_functions.DefaultEmbeddingFunction()

    try:
        collection = client.get_collection(
            name='retail_intelligence',
            embedding_function=ef
        )
        query   = f"Store {state['store_id']} demand forecast stockout risk inventory"
        results = collection.query(query_texts=[query], n_results=3)
        rag_context = '\n'.join(results['documents'][0])
    except Exception:
        rag_context = "No additional context available from knowledge base."

    # Build prompt
    forecast   = state.get('forecast', {})
    shap       = state.get('shap_explanation', {})
    inventory  = state.get('inventory_decision', {})
    critic     = state.get('critic_score', {})

    prompt = f"""
You are a retail strategy consultant. Based on the analysis below, write a concise
executive brief for Store {state['store_id']}.

FORECAST ANALYSIS:
- 4-week demand forecast: {forecast.get('forecast_4wk', [])}
- Average forecast: ${forecast.get('avg_forecast', 0):,.0f}/week
- Trend: {forecast.get('trend', 'unknown')}

SHAP EXPLAINABILITY:
- Top demand driver: {shap.get('top_driver', 'unknown')}
- Holiday week impact: ${shap.get('holiday_impact', 0):,.0f}
- {shap.get('summary_text', '')}

INVENTORY DECISION (confidence: {inventory.get('confidence', 0):.0%}):
- Risk level: {inventory.get('risk_level', 'unknown')}
- Recommended reorder: {inventory.get('reorder_qty', 0):,.0f} units
- High-risk departments: {inventory.get('risk_depts', [])}
- {inventory.get('justification', '')}

QUALITY CHECK: Score {critic.get('score', 0):.0%} — {critic.get('feedback', '')}

KNOWLEDGE BASE CONTEXT:
{rag_context}

Write your response as a JSON object with these exact keys:
- executive_summary: 2-3 sentence summary
- key_findings: list of 3 specific findings with dollar amounts
- recommended_actions: list of 3 concrete actions
- dollar_impact: estimated annual impact of recommendations
"""

    llm = ChatGroq(
        model_name  ='llama-3.3-70b-versatile',
        temperature =0.2,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You are a retail strategy consultant. Always respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])

        raw = response.content.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]

        brief_data = json.loads(raw)

        output = StrategyBrief(
            store_id            = state['store_id'],
            executive_summary   = brief_data.get('executive_summary', ''),
            key_findings        = brief_data.get('key_findings', []),
            recommended_actions = brief_data.get('recommended_actions', []),
            dollar_impact       = brief_data.get('dollar_impact', 'Not quantified')
        )

    except Exception as e:
        print(f"  LLM error: {e} — using fallback brief")
        output = StrategyBrief(
            store_id            = state['store_id'],
            executive_summary   = f"Store {state['store_id']} shows {forecast.get('trend','stable')} demand trend with {inventory.get('risk_level','MEDIUM')} stockout risk.",
            key_findings        = [
                f"Top driver: {shap.get('top_driver','lag_1')} with ${shap.get('top_driver_value',0):,.0f} SHAP impact",
                f"Holiday weeks add ${shap.get('holiday_impact',0):,.0f} to weekly sales",
                f"Recommended reorder: {inventory.get('reorder_qty',0):,.0f} units across top departments"
            ],
            recommended_actions = [
                f"Replenish departments {inventory.get('risk_depts',[])} immediately",
                "Increase safety stock by 20% ahead of next holiday week",
                "Monitor CPI and fuel price trends for demand sensitivity"
            ],
            dollar_impact = f"Estimated ${inventory.get('reorder_qty',0)*0.15:,.0f} in prevented stockout losses"
        )

    print(f"  Brief generated for Store {state['store_id']}")
    print(f"  Summary: {output.executive_summary[:100]}...")

    state['strategy_brief'] = output.dict()
    return state
