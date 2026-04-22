import json
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from agents.nodes import (
    forecast_agent,
    shap_explainer_agent,
    inventory_optimizer_agent,
    critic_agent,
    critic_router,
    executive_strategist_agent
)
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent
load_dotenv(BASE_PATH / '.env')


# ── State schema ──────────────────────────────────────────────────────────────

class RetailState(TypedDict):
    store_id            : int
    dept                : int
    forecast            : Optional[dict]
    shap_explanation    : Optional[dict]
    inventory_decision  : Optional[dict]
    critic_score        : Optional[dict]
    strategy_brief      : Optional[dict]
    retry_count         : int
    error               : Optional[str]
    user_query          : Optional[str]


# ── Build pipeline ────────────────────────────────────────────────────────────

def build_pipeline():
    graph = StateGraph(RetailState)

    # Add nodes
    graph.add_node('forecast_agent',           forecast_agent)
    graph.add_node('shap_explainer',           shap_explainer_agent)
    graph.add_node('inventory_optimizer',      inventory_optimizer_agent)
    graph.add_node('critic_agent',             critic_agent)
    graph.add_node('executive_strategist',     executive_strategist_agent)

    # Linear edges
    graph.set_entry_point('forecast_agent')
    graph.add_edge('forecast_agent',       'shap_explainer')
    graph.add_edge('shap_explainer',       'inventory_optimizer')
    graph.add_edge('inventory_optimizer',  'critic_agent')

    # Conditional edge — reflection loop
    graph.add_conditional_edges(
        'critic_agent',
        critic_router,
        {
            'inventory_optimizer' : 'inventory_optimizer',
            'executive_strategist': 'executive_strategist'
        }
    )

    graph.add_edge('executive_strategist', END)

    return graph.compile()


# ── Run pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(store_id: int, dept: int, user_query: str = '') -> dict:
    pipeline = build_pipeline()

    initial_state = RetailState(
        store_id           = store_id,
        dept               = dept,
        forecast           = None,
        shap_explanation   = None,
        inventory_decision = None,
        critic_score       = None,
        strategy_brief     = None,
        retry_count        = 0,
        error              = None,
        user_query         = user_query
    )

    print(f"\n{'='*60}")
    print(f"Running pipeline — Store {store_id}, Dept {dept}")
    print(f"{'='*60}")

    result = pipeline.invoke(initial_state)

    print(f"\n{'='*60}")
    print("Pipeline complete")
    print(f"{'='*60}")

    return result


if __name__ == '__main__':
    # Run ChromaDB ingestion first
    from agents.chroma_ingest import ingest_shap_summaries
    print("Ingesting SHAP summaries into ChromaDB...")
    ingest_shap_summaries()

    # Run pipeline for Store 1, Dept 1
    result = run_pipeline(store_id=1, dept=1, user_query="What is the stockout risk for Store 1?")

    # Print final brief
    brief = result.get('strategy_brief', {})
    print(f"\n{'='*60}")
    print("EXECUTIVE BRIEF")
    print(f"{'='*60}")
    print(f"\nSummary:\n{brief.get('executive_summary','')}")
    print(f"\nKey findings:")
    for f in brief.get('key_findings', []):
        print(f"  - {f}")
    print(f"\nRecommended actions:")
    for a in brief.get('recommended_actions', []):
        print(f"  - {a}")
    print(f"\nDollar impact: {brief.get('dollar_impact','')}")

    # Save result
    result_path = BASE_PATH / 'data' / 'processed' / 'pipeline_result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull result saved → data/processed/pipeline_result.json")
