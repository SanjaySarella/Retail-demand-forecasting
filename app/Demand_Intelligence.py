import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from agents.pipeline import run_pipeline
from agents.chroma_ingest import ingest_shap_summaries

st.set_page_config(
    page_title="Retail Demand Intelligence",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .risk-high   { color: #D85A30; font-weight: 600; }
    .risk-medium { color: #BA7517; font-weight: 600; }
    .risk-low    { color: #1D9E75; font-weight: 600; }
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    path = os.path.join(BASE_PATH, 'data', 'processed', 'walmart_clean.csv')
    return pd.read_csv(path, parse_dates=['Date'])

@st.cache_data
def load_predictions():
    path = os.path.join(BASE_PATH, 'data', 'processed', 'predictions.csv')
    return pd.read_csv(path, parse_dates=['Date'])

@st.cache_data
def load_shap():
    path = os.path.join(BASE_PATH, 'data', 'processed', 'shap_values.csv')
    return pd.read_csv(path, parse_dates=['Date'])

@st.cache_data
def load_shap_summary():
    path = os.path.join(BASE_PATH, 'data', 'processed', 'shap_summary.json')
    with open(path, 'r') as f:
        return json.load(f)
    
def answer_query(query: str, store_id: int, dept_id: int) -> str:
    """
    Answers a specific user question using ChromaDB RAG + Groq.
    Actually reads the question and answers it directly.
    """
    import chromadb
    from chromadb.utils import embedding_functions
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    # Pull relevant context from ChromaDB
    chroma_path = os.path.join(BASE_PATH, 'data', 'processed', 'chromadb')
    try:
        client     = chromadb.PersistentClient(path=chroma_path)
        ef         = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_collection(
            name='retail_intelligence',
            embedding_function=ef
        )
        results     = collection.query(query_texts=[query], n_results=3)
        rag_context = '\n\n'.join(results['documents'][0])
    except Exception:
        rag_context = "No knowledge base context available."

    # Pull live data context for the selected store
    try:
        df         = load_data()
        store_data = df[df['Store'] == store_id]
        avg_sales  = store_data['Weekly_Sales'].mean()
        max_sales  = store_data['Weekly_Sales'].max()
        holiday_avg = store_data[store_data['IsHoliday'] == True]['Weekly_Sales'].mean()
        nonholiday_avg = store_data[store_data['IsHoliday'] == False]['Weekly_Sales'].mean()
        holiday_lift = ((holiday_avg - nonholiday_avg) / nonholiday_avg * 100)
        store_type = store_data['Type'].iloc[0]
        top_depts  = (
            store_data.groupby('Dept')['Weekly_Sales']
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        data_context = f"""
Store {store_id} data summary:
- Store type: {store_type}
- Average weekly sales: ${avg_sales:,.0f}
- Peak weekly sales: ${max_sales:,.0f}
- Holiday week avg: ${holiday_avg:,.0f} (+{holiday_lift:.1f}% vs non-holiday)
- Top 5 departments by sales: {top_depts}
- Selected department: {dept_id}
"""
    except Exception:
        data_context = f"Store {store_id} data not available."

    # Load SHAP context
    try:
        shap_data    = load_shap_summary()
        top_feature  = shap_data.get('top_feature', 'lag_1')
        holiday_shap = shap_data.get('holiday_avg_shap', 0)
        store_shap   = next(
            (s for s in shap_data.get('store_summaries', []) if s['store_id'] == store_id),
            None
        )
        shap_context = f"""
SHAP analysis for Store {store_id}:
- Top global demand driver: {top_feature}
- Holiday SHAP contribution: ${holiday_shap:,.0f} per holiday week
"""
        if store_shap:
            shap_context += f"- Store-specific top drivers: {[d[0] for d in store_shap['top_3_drivers']]}\n"
            shap_context += f"- Store markdown impact: ${store_shap['markdown_impact']:,.0f}\n"
    except Exception:
        shap_context = "SHAP data not available."

    # Load pipeline result if available
    pipeline_context = ""
    if st.session_state.pipeline_result:
        result   = st.session_state.pipeline_result
        inv      = result.get('inventory_decision', {})
        forecast = result.get('forecast', {})
        pipeline_context = f"""
Latest pipeline results for Store {store_id}:
- 4-week demand forecast: {forecast.get('forecast_4wk', [])}
- Demand trend: {forecast.get('trend', 'unknown')}
- Risk level: {inv.get('risk_level', 'unknown')}
- High-risk departments: {inv.get('risk_depts', [])}
- Recommended reorder qty: {inv.get('reorder_qty', 0):,.0f} units
- Justification: {inv.get('justification', '')}
"""

    # Build conversation history for context
    history_text = ""
    if len(st.session_state.chat_history) > 1:
        recent = st.session_state.chat_history[-4:]
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Call Groq with full context
    groq_key = None
    env_path = os.path.join(BASE_PATH, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('GROQ_API_KEY'):
                    groq_key = line.strip().split('=', 1)[1]
                    break

    if not groq_key:
        groq_key = os.getenv('GROQ_API_KEY', '')

    try:
        llm = ChatGroq(
            model_name   = 'llama-3.3-70b-versatile',
            temperature  = 0.3,
            groq_api_key = groq_key
        )

        system_prompt = """You are a retail strategy AI consultant with deep expertise in 
demand forecasting, inventory management, and Walmart store operations.
You have access to real store data, SHAP analysis, and forecast results.
Answer questions specifically and concisely. Use dollar amounts and percentages 
where available. Never give a generic answer — always reference the specific 
store, department, or data point the user is asking about.
Keep answers under 200 words unless the question requires more detail."""

        user_prompt = f"""
STORE DATA:
{data_context}

SHAP EXPLAINABILITY:
{shap_context}

PIPELINE RESULTS:
{pipeline_context if pipeline_context else "Run the pipeline first for live forecast data."}

KNOWLEDGE BASE:
{rag_context}

CONVERSATION HISTORY:
{history_text if history_text else "This is the start of the conversation."}

USER QUESTION: {query}

Answer the user's specific question using the data above.
"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content

    except Exception as e:
        return (
            f"I encountered an error connecting to the AI model: {str(e)}\n\n"
            f"Based on available data — Store {store_id} averages ${avg_sales:,.0f}/week "
            f"with a {holiday_lift:.1f}% holiday sales lift. "
            f"Top departments: {top_depts}."
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Retail demand intelligence")
    st.markdown("*Walmart 45-store forecast system*")
    st.divider()

    store_id = st.selectbox(
        "Select store",
        options=list(range(1, 46)),
        index=0
    )

    df_raw = load_data()
    depts  = sorted(df_raw[df_raw['Store'] == store_id]['Dept'].unique().tolist())

    dept_id = st.selectbox(
        "Select department",
        options=depts,
        index=0
    )

    st.divider()
    st.markdown("### Run AI pipeline")
    run_button = st.button("Analyse this store", use_container_width=True, type="primary")


# ── Session state ─────────────────────────────────────────────────────────────

if 'pipeline_result' not in st.session_state:
    st.session_state.pipeline_result = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if run_button:
    with st.spinner(f"Running 5-node pipeline for Store {store_id}..."):
        result = run_pipeline(
            store_id   = store_id,
            dept       = dept_id,
            user_query = ''
        )
        st.session_state.pipeline_result = result
    st.success("Pipeline complete")


# ── Main dashboard ────────────────────────────────────────────────────────────

st.markdown("# Retail demand forecasting — intelligence dashboard")
st.markdown(f"**Store {store_id}** · Department {dept_id} · Walmart 45-store dataset (2010–2012)")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecast vs actual",
    "Store risk heatmap",
    "Holiday impact",
    "SHAP feature importance",
    "AI strategy brief"
])


# ── Tab 1: Forecast vs actual ─────────────────────────────────────────────────

with tab1:
    st.markdown("### Weekly demand forecast vs actual sales")

    try:
        preds = load_predictions()
        store_preds = preds[
            (preds['Store'] == store_id) &
            (preds['Dept']  == dept_id)
        ].sort_values('Date')

        if not store_preds.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=store_preds['Date'],
                y=store_preds['Weekly_Sales'],
                name='Actual',
                line=dict(color='#185FA5', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=store_preds['Date'],
                y=store_preds['XGB_Predicted'],
                name='XGBoost Forecast',
                line=dict(color='#D85A30', width=2, dash='dash')
            ))
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Weekly Sales ($)',
                legend=dict(orientation='h', y=1.1),
                height=400,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            mae          = np.abs(store_preds['Residual']).mean()
            mape         = store_preds['Error_Pct'].mean()
            avg_actual   = store_preds['Weekly_Sales'].mean()
            avg_forecast = store_preds['XGB_Predicted'].mean()

            col1.metric("Avg actual sales", f"${avg_actual:,.0f}")
            col2.metric("Avg forecast",     f"${avg_forecast:,.0f}")
            col3.metric("MAE",              f"${mae:,.0f}")
            col4.metric("MAPE",             f"{mape:.1f}%")
        else:
            st.info("No prediction data for this store-dept combination.")
    except Exception as e:
        st.warning(f"Load predictions first by running the pipeline. ({e})")


# ── Tab 2: Store risk heatmap ─────────────────────────────────────────────────

with tab2:
    st.markdown("### 45-store stockout risk heatmap")

    try:
        preds = load_predictions()

        store_risk = preds.groupby('Store').apply(lambda x: pd.Series({
            'Avg_Sales'    : x['Weekly_Sales'].mean(),
            'Avg_Forecast' : x['XGB_Predicted'].mean(),
            'Avg_Error_Pct': x['Error_Pct'].mean(),
            'Holiday_Weeks': x['IsHoliday'].sum()
        })).reset_index()

        store_risk['Risk_Score'] = (
            (store_risk['Avg_Forecast'] / store_risk['Avg_Sales']) *
            (1 + store_risk['Avg_Error_Pct'] / 100)
        ).round(3)

        store_risk['Risk_Level'] = store_risk['Risk_Score'].apply(
            lambda x: 'HIGH' if x > 1.3 else ('MEDIUM' if x > 1.1 else 'LOW')
        )

        color_map = {'HIGH': '#D85A30', 'MEDIUM': '#BA7517', 'LOW': '#1D9E75'}

        fig = px.scatter(
            store_risk,
            x='Avg_Sales',
            y='Avg_Forecast',
            size='Avg_Error_Pct',
            color='Risk_Level',
            color_discrete_map=color_map,
            hover_data=['Store','Risk_Score','Holiday_Weeks'],
            text='Store',
            title='Store risk positioning — actual vs forecast demand',
            labels={
                'Avg_Sales'   : 'Avg actual weekly sales ($)',
                'Avg_Forecast': 'Avg forecast weekly sales ($)'
            }
        )
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Risk summary")
        col1, col2, col3 = st.columns(3)
        high_count   = (store_risk['Risk_Level'] == 'HIGH').sum()
        medium_count = (store_risk['Risk_Level'] == 'MEDIUM').sum()
        low_count    = (store_risk['Risk_Level'] == 'LOW').sum()
        col1.metric("High risk stores",   high_count)
        col2.metric("Medium risk stores", medium_count)
        col3.metric("Low risk stores",    low_count)

        st.dataframe(
            store_risk[['Store','Avg_Sales','Avg_Forecast','Risk_Score','Risk_Level']]
            .sort_values('Risk_Score', ascending=False)
            .round(2),
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Run the pipeline first to generate predictions. ({e})")


# ── Tab 3: Holiday impact ─────────────────────────────────────────────────────

with tab3:
    st.markdown("### Holiday week sales impact analysis")

    try:
        df_raw = load_data()

        holiday_summary = df_raw.groupby(
            ['Date','IsHoliday'])['Weekly_Sales'].mean().reset_index()
        holiday_avg = df_raw.groupby('IsHoliday')['Weekly_Sales'].mean()
        lift_pct    = (
            (holiday_avg[True] - holiday_avg[False]) / holiday_avg[False] * 100
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Non-holiday avg",    f"${holiday_avg[False]:,.0f}")
        col2.metric("Holiday avg",        f"${holiday_avg[True]:,.0f}")
        col3.metric("Holiday sales lift", f"+{lift_pct:.1f}%")

        fig = px.line(
            holiday_summary.sort_values('Date'),
            x='Date',
            y='Weekly_Sales',
            color='IsHoliday',
            color_discrete_map={False: '#185FA5', True: '#D85A30'},
            labels={
                'Weekly_Sales': 'Avg weekly sales ($)',
                'IsHoliday'   : 'Holiday week'
            },
            title='Average weekly sales — holiday vs non-holiday weeks'
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Holiday week breakdown")
        holiday_weeks = df_raw[df_raw['IsHoliday'] == True].copy()
        holiday_weeks['Holiday_Name'] = holiday_weeks['Date'].apply(lambda d:
            'Super Bowl'   if d.month == 2  else
            'Labor Day'    if d.month == 9  else
            'Thanksgiving' if d.month == 11 else
            'Christmas'    if d.month == 12 else 'Other'
        )
        holiday_by_event = (
            holiday_weeks.groupby('Holiday_Name')['Weekly_Sales']
            .mean()
            .sort_values(ascending=False)
        )

        fig2 = px.bar(
            holiday_by_event.reset_index(),
            x='Holiday_Name',
            y='Weekly_Sales',
            color='Holiday_Name',
            color_discrete_sequence=['#185FA5','#1D9E75','#D85A30','#BA7517'],
            labels={
                'Weekly_Sales': 'Avg weekly sales ($)',
                'Holiday_Name': 'Holiday'
            },
            title='Average sales by holiday event'
        )
        fig2.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── Tab 4: SHAP feature importance ───────────────────────────────────────────

with tab4:
    st.markdown("### SHAP feature importance — what drives sales")

    try:
        shap_summary = load_shap_summary()
        top10        = pd.DataFrame(shap_summary['global_top10_features'])

        fig = px.bar(
            top10.sort_values('mean_abs_shap'),
            x='mean_abs_shap',
            y='feature',
            orientation='h',
            color='mean_abs_shap',
            color_continuous_scale=['#E6F1FB','#185FA5'],
            labels={
                'mean_abs_shap': 'Mean |SHAP| value',
                'feature'      : 'Feature'
            },
            title='Top 10 features by mean absolute SHAP value'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Holiday SHAP impact")
            st.metric(
                "Avg holiday contribution",
                f"${shap_summary['holiday_avg_shap']:,.0f}",
                delta="per holiday week"
            )
            st.markdown(f"**Top global driver:** `{shap_summary['top_feature']}`")

        with col2:
            st.markdown("#### Store-level SHAP summaries")
            store_sum = shap_summary.get('store_summaries', [])
            if store_sum:
                for s in store_sum[:3]:
                    with st.expander(f"Store {s['store_id']}"):
                        st.write(f"Top drivers: {', '.join([d[0] for d in s['top_3_drivers']])}")
                        st.write(f"Holiday impact: ${s['holiday_impact']:,.0f}")
                        st.write(f"Markdown impact: ${s['markdown_impact']:,.0f}")

    except Exception as e:
        st.warning(f"Run SHAP notebook first. ({e})")


# ── Tab 5: AI strategy brief ──────────────────────────────────────────────────

with tab5:
    st.markdown("### AI-generated executive strategy brief")

    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        brief  = result.get('strategy_brief', {})
        inv    = result.get('inventory_decision', {})
        critic = result.get('critic_score', {})
        fore   = result.get('forecast', {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg weekly forecast", f"${fore.get('avg_forecast',0):,.0f}")
        col2.metric("Demand trend",        fore.get('trend','—').capitalize())
        col3.metric("Risk level",          inv.get('risk_level','—'))
        col4.metric("Critic score",        f"{critic.get('score',0):.0%}")

        st.divider()

        st.markdown("#### Executive summary")
        st.info(brief.get('executive_summary', 'Run the pipeline to generate.'))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Key findings")
            for finding in brief.get('key_findings', []):
                st.markdown(f"- {finding}")

        with col2:
            st.markdown("#### Recommended actions")
            for action in brief.get('recommended_actions', []):
                st.markdown(f"- {action}")

        st.divider()
        st.markdown("#### Dollar impact")
        st.success(brief.get('dollar_impact', 'Not quantified'))

        st.divider()
        st.markdown("#### Purchase order draft")
        po = inv.get('purchase_order', {})
        if po.get('line_items'):
            po_df = pd.DataFrame(po['line_items'])
            st.dataframe(po_df, use_container_width=True)
            st.markdown(f"**Total estimated value:** ${po.get('total_estimated_value',0):,.0f}")
            st.markdown(f"**Recommended action:** {po.get('recommended_action','')}")

    else:
        st.info("Select a store from the sidebar and click **Analyse this store** to generate the AI brief.")

    st.divider()
    st.markdown("#### Chat with the AI agent")

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

