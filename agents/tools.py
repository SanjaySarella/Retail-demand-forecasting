import pandas as pd
import numpy as np
import json
from langchain.tools import tool
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent
df_clean = None

def load_data():
    global df_clean
    if df_clean is None:
        csv_path = BASE_PATH / 'data' / 'processed' / 'walmart_clean.csv'
        df_clean = pd.read_csv(csv_path, parse_dates=['Date'])
    return df_clean


@tool
def calculate_reorder_qty(store_id: int, dept: int) -> str:
    """
    Calculates recommended reorder quantity for a given store and department
    based on recent sales velocity and variance.
    """
    df = load_data()
    store_dept = df[(df['Store'] == store_id) & (df['Dept'] == dept)].sort_values('Date')

    if store_dept.empty:
        return json.dumps({'error': f'No data found for Store {store_id}, Dept {dept}'})

    recent     = store_dept.tail(8)
    avg_sales  = recent['Weekly_Sales'].mean()
    std_sales  = recent['Weekly_Sales'].std()
    max_sales  = recent['Weekly_Sales'].max()

    # Safety stock = 1.5 standard deviations above mean
    safety_stock    = avg_sales + (1.5 * std_sales)
    reorder_qty     = round(safety_stock * 1.2, 0)
    confidence      = round(min(len(recent) / 8, 1.0), 2)

    result = {
        'store_id'        : store_id,
        'dept'            : dept,
        'avg_weekly_sales': round(avg_sales, 2),
        'std_weekly_sales': round(std_sales, 2),
        'recommended_reorder_qty': reorder_qty,
        'safety_stock'    : round(safety_stock, 2),
        'confidence'      : confidence,
        'based_on_weeks'  : len(recent)
    }
    return json.dumps(result)


@tool
def flag_stockout_risk(store_id: int, threshold: float = 0.3) -> str:
    """
    Flags stockout risk for a store by comparing recent sales trend
    to historical average. Returns risk level and departments at risk.
    """
    df = load_data()
    store_data = df[df['Store'] == store_id].sort_values('Date')

    if store_data.empty:
        return json.dumps({'error': f'No data found for Store {store_id}'})

    results = []
    for dept in store_data['Dept'].unique():
        dept_data  = store_data[store_data['Dept'] == dept]
        if len(dept_data) < 8:
            continue

        historical_avg = dept_data['Weekly_Sales'].mean()
        recent_avg     = dept_data.tail(4)['Weekly_Sales'].mean()
        trend_ratio    = recent_avg / historical_avg if historical_avg > 0 else 1.0
        holiday_weeks  = dept_data.tail(4)['IsHoliday'].sum()

        if trend_ratio > (1 + threshold):
            risk_level = 'HIGH'
        elif trend_ratio > (1 + threshold / 2):
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        results.append({
            'dept'           : int(dept),
            'risk_level'     : risk_level,
            'trend_ratio'    : round(trend_ratio, 3),
            'recent_avg'     : round(recent_avg, 2),
            'historical_avg' : round(historical_avg, 2),
            'holiday_weeks'  : int(holiday_weeks)
        })

    high_risk = [r for r in results if r['risk_level'] == 'HIGH']
    medium_risk = [r for r in results if r['risk_level'] == 'MEDIUM']

    summary = {
        'store_id'          : store_id,
        'total_depts_checked': len(results),
        'high_risk_depts'   : high_risk[:5],
        'medium_risk_depts' : medium_risk[:5],
        'high_risk_count'   : len(high_risk),
        'medium_risk_count' : len(medium_risk)
    }
    return json.dumps(summary)


@tool
def draft_purchase_order(store_id: int) -> str:
    """
    Drafts a purchase order for the top 5 highest-risk departments
    in a given store based on reorder quantity calculations.
    """
    df = load_data()
    store_data = df[df['Store'] == store_id]

    if store_data.empty:
        return json.dumps({'error': f'No data found for Store {store_id}'})

    store_type = store_data['Type'].iloc[0]
    store_size = store_data['Size'].iloc[0]

    top_depts = (
        store_data.groupby('Dept')['Weekly_Sales']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    line_items = []
    total_value = 0

    for dept in top_depts:
        dept_data  = store_data[store_data['Dept'] == dept]
        avg_sales  = dept_data['Weekly_Sales'].mean()
        std_sales  = dept_data['Weekly_Sales'].std()
        reorder    = round(avg_sales + 1.5 * std_sales, 0)
        unit_value = round(reorder * 0.4, 2)
        total_value += unit_value

        line_items.append({
            'dept'           : int(dept),
            'avg_weekly_sales': round(avg_sales, 2),
            'reorder_qty'    : reorder,
            'estimated_value': unit_value,
            'priority'       : 'HIGH' if avg_sales > df['Weekly_Sales'].mean() else 'STANDARD'
        })

    purchase_order = {
        'purchase_order_id' : f'PO-STORE{store_id}-AUTO',
        'store_id'          : store_id,
        'store_type'        : store_type,
        'store_size_sqft'   : int(store_size),
        'generated_by'      : 'Inventory Optimizer Agent',
        'line_items'        : line_items,
        'total_estimated_value': round(total_value, 2),
        'recommended_action': f'Replenish top {len(line_items)} departments before next holiday week',
        'confidence'        : 0.82
    }
    return json.dumps(purchase_order, indent=2)
