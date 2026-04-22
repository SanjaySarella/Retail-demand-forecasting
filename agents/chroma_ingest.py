import json
import os
import chromadb
from chromadb.utils import embedding_functions
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent
CHROMA_PATH = str(BASE_PATH / 'data' / 'processed' / 'chromadb')


def ingest_shap_summaries():
    """Ingest SHAP summaries into ChromaDB for RAG retrieval."""

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name='retail_intelligence',
        embedding_function=ef
    )

    # Load SHAP summary JSON
    shap_path = BASE_PATH / 'data' / 'processed' / 'shap_summary.json'
    with open(shap_path, 'r') as f:
        shap_data = json.load(f)

    documents = []
    metadatas = []
    ids       = []

    # Global summary document
    global_doc = f"""
    Global retail demand forecast analysis across 45 Walmart stores.
    Top feature driving sales: {shap_data['top_feature']}.
    Holiday weeks contribute an average SHAP value of ${shap_data['holiday_avg_shap']:,.0f} to weekly sales.
    Top 10 features by importance: {', '.join([f['feature'] for f in shap_data['global_top10_features']])}.
    """
    documents.append(global_doc.strip())
    metadatas.append({'type': 'global_summary', 'source': 'shap_analysis'})
    ids.append('global_summary_001')

    # Store-level summaries
    for store in shap_data.get('store_summaries', []):
        store_id  = store['store_id']
        top3      = store['top_3_drivers']
        holiday   = store['holiday_impact']
        markdown  = store['markdown_impact']

        store_doc = f"""
        Store {store_id} demand forecast analysis.
        Top 3 sales drivers: {', '.join([f[0] for f in top3])}.
        Holiday impact: ${holiday:,.0f} average SHAP contribution per holiday week.
        Markdown promotions impact: ${markdown:,.0f} average SHAP contribution.
        Lag features (recent sales history) are among the strongest predictors for this store.
        """
        documents.append(store_doc.strip())
        metadatas.append({'type': 'store_summary', 'store_id': store_id})
        ids.append(f'store_summary_{store_id:03d}')

    # Retail strategy knowledge base documents
    strategy_docs = [
        {
            'id'  : 'strategy_001',
            'text': """
                Stockout risk management in retail. When demand forecast exceeds
                historical average by more than 30%, immediate reorder action is
                recommended. Holiday weeks (Super Bowl, Thanksgiving, Christmas)
                historically show 15-40% sales spikes. Safety stock calculation:
                average demand + 1.5 standard deviations ensures 93% service level.
                """,
            'meta': {'type': 'strategy', 'topic': 'stockout_management'}
        },
        {
            'id'  : 'strategy_002',
            'text': """
                Inventory optimization principles for large format retail stores.
                Type A stores (largest format) carry broader assortment and require
                higher safety stock levels. Type B stores balance depth and breadth.
                Type C stores focus on fast-moving items. Department-level forecasting
                outperforms store-level forecasting by 15-20% in accuracy.
                """,
            'meta': {'type': 'strategy', 'topic': 'inventory_optimization'}
        },
        {
            'id'  : 'strategy_003',
            'text': """
                Markdown and promotional impact on retail demand forecasting.
                Markdown events (MarkDown1 through MarkDown5) represent different
                promotional categories. Overlapping markdowns can cause demand spikes
                of 20-50%. Models must account for markdown interaction effects.
                Post-promotion dip effect typically reduces demand 10-15% in the
                two weeks following a major markdown event.
                """,
            'meta': {'type': 'strategy', 'topic': 'markdown_impact'}
        },
        {
            'id'  : 'strategy_004',
            'text': """
                Macroeconomic factors in retail demand. CPI (Consumer Price Index)
                and unemployment rate are lagging indicators that influence consumer
                spending patterns. High unemployment reduces discretionary spending.
                Fuel price increases reduce store visit frequency. Temperature
                affects seasonal category performance — cold weather drives heating
                and winter apparel, warm weather drives outdoor and garden categories.
                """,
            'meta': {'type': 'strategy', 'topic': 'macro_factors'}
        }
    ]

    for doc in strategy_docs:
        documents.append(doc['text'].strip())
        metadatas.append(doc['meta'])
        ids.append(doc['id'])

    # Add all to ChromaDB
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"ChromaDB ingestion complete")
    print(f"Collection: retail_intelligence")
    print(f"Documents ingested: {len(documents)}")
    print(f"Stored at: {CHROMA_PATH}")

    return collection


if __name__ == '__main__':
    ingest_shap_summaries()
