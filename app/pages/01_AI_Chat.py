import streamlit as st
import os
import sys
import json

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

st.set_page_config(
    page_title="AI Retail Agent",
    page_icon="🤖",
    layout="centered"
)

st.markdown("# Retail AI agent")
st.markdown("Ask anything about store performance, stockout risk, holiday impact, or inventory strategy.")
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    path = os.path.join(BASE_PATH, 'data', 'processed', 'walmart_clean.csv')
    return pd.read_csv(path, parse_dates=['Date'])

@st.cache_data
def load_shap_summary():
    path = os.path.join(BASE_PATH, 'data', 'processed', 'shap_summary.json')
    with open(path, 'r') as f:
        return json.load(f)

def get_groq_key():
    env_path = os.path.join(BASE_PATH, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('GROQ_API_KEY'):
                    return line.strip().split('=', 1)[1]
    return os.getenv('GROQ_API_KEY', '')

def get_rag_context(query: str) -> str:
    chroma_path = os.path.join(BASE_PATH, 'data', 'processed', 'chromadb')
    try:
        client     = chromadb.PersistentClient(path=chroma_path)
        ef         = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_collection(
            name='retail_intelligence',
            embedding_function=ef
        )
        results = collection.query(query_texts=[query], n_results=3)
        return '\n\n'.join(results['documents'][0])
    except Exception:
        return "Knowledge base not available."

def get_store_context(store_id: int) -> str:
    try:
        df         = load_data()
        store_data = df[df['Store'] == store_id]
        avg_sales  = store_data['Weekly_Sales'].mean()
        max_sales  = store_data['Weekly_Sales'].max()
        holiday_avg    = store_data[store_data['IsHoliday'] == True]['Weekly_Sales'].mean()
        nonholiday_avg = store_data[store_data['IsHoliday'] == False]['Weekly_Sales'].mean()
        holiday_lift   = ((holiday_avg - nonholiday_avg) / nonholiday_avg * 100)
        store_type = store_data['Type'].iloc[0]
        top_depts  = (
            store_data.groupby('Dept')['Weekly_Sales']
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        return f"""
Store {store_id} ({store_type} type):
- Avg weekly sales: ${avg_sales:,.0f}
- Peak weekly sales: ${max_sales:,.0f}
- Holiday lift: +{holiday_lift:.1f}%
- Top 5 departments: {top_depts}
"""
    except Exception:
        return f"Store {store_id} data not available."

def get_shap_context(store_id: int) -> str:
    try:
        shap_data   = load_shap_summary()
        top_feature = shap_data.get('top_feature', 'lag_1')
        holiday_shap= shap_data.get('holiday_avg_shap', 0)
        store_shap  = next(
            (s for s in shap_data.get('store_summaries', []) if s['store_id'] == store_id),
            None
        )
        text = f"Top demand driver: {top_feature}. Holiday SHAP: ${holiday_shap:,.0f}/week.\n"
        if store_shap:
            text += f"Store {store_id} top drivers: {[d[0] for d in store_shap['top_3_drivers']]}.\n"
            text += f"Markdown impact: ${store_shap['markdown_impact']:,.0f}.\n"
        return text
    except Exception:
        return "SHAP data not available."

def answer(query: str, store_id: int, history: list) -> str:
    rag     = get_rag_context(query)
    store   = get_store_context(store_id)
    shap    = get_shap_context(store_id)

    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    system = """You are a retail strategy AI consultant specialising in 
demand forecasting and inventory management for large-format retail stores.
You have access to real Walmart store data, SHAP explainability analysis, 
and a retail strategy knowledge base.
Be specific, concise, and always reference actual numbers from the data.
Never give a generic answer. Keep responses under 200 words unless more detail is needed."""

    prompt = f"""
STORE DATA:
{store}

SHAP ANALYSIS:
{shap}

KNOWLEDGE BASE:
{rag}

CONVERSATION HISTORY:
{history_text if history_text else "Start of conversation."}

USER QUESTION: {query}

Answer specifically using the data above.
"""

    try:
        llm = ChatGroq(
            model_name   ='llama-3.3-70b-versatile',
            temperature  =0.3,
            groq_api_key =get_groq_key()
        )
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt)
        ])
        return response.content
    except Exception as e:
        return f"Error connecting to AI model: {str(e)}"


# ── Store selector ────────────────────────────────────────────────────────────

store_id = st.selectbox(
    "Which store are you asking about?",
    options=list(range(1, 46)),
    index=0,
    key="chat_store"
)

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            'role'   : 'assistant',
            'content': f"Hi! I'm your retail strategy agent. I have full access to sales data for all 45 stores, SHAP analysis, and inventory intelligence. Ask me anything — stockout risk, holiday impact, department performance, reorder strategy."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about any store, department, or retail strategy..."):
    st.session_state.messages.append({
        'role'   : 'user',
        'content': prompt
    })
    with st.chat_message('user'):
        st.write(prompt)

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = answer(
                query    = prompt,
                store_id = store_id,
                history  = st.session_state.messages[:-1]
            )
        st.write(response)

    st.session_state.messages.append({
        'role'   : 'assistant',
        'content': response
    })

# ── Quick question buttons ────────────────────────────────────────────────────

st.divider()
st.markdown("#### Quick questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Which depts are at stockout risk?", use_container_width=True):
        q = f"Which departments in Store {store_id} are at highest stockout risk and why?"
        st.session_state.messages.append({'role': 'user', 'content': q})
        with st.spinner("Thinking..."):
            r = answer(q, store_id, st.session_state.messages[:-1])
        st.session_state.messages.append({'role': 'assistant', 'content': r})
        st.rerun()

with col2:
    if st.button("What is the holiday sales impact?", use_container_width=True):
        q = f"What is the holiday week sales impact for Store {store_id}? Give specific numbers."
        st.session_state.messages.append({'role': 'user', 'content': q})
        with st.spinner("Thinking..."):
            r = answer(q, store_id, st.session_state.messages[:-1])
        st.session_state.messages.append({'role': 'assistant', 'content': r})
        st.rerun()

with col3:
    if st.button("What should I reorder this week?", use_container_width=True):
        q = f"What should Store {store_id} prioritise for reordering this week based on sales trends?"
        st.session_state.messages.append({'role': 'user', 'content': q})
        with st.spinner("Thinking..."):
            r = answer(q, store_id, st.session_state.messages[:-1])
        st.session_state.messages.append({'role': 'assistant', 'content': r})
        st.rerun()

st.divider()
if st.button("Clear chat", use_container_width=False):
    st.session_state.messages = [
        {
            'role'   : 'assistant',
            'content': "Chat cleared. Ask me anything about store performance, inventory, or demand forecasting."
        }
    ]
    st.rerun()