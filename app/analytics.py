"""Historical sales analysis. All forecasts are relative to the dataset cutoff."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data/raw/train.csv'
METHODS = ('Last observed week', 'Same week 52 weeks earlier')

def load_sales(path=DATA):
    df = pd.read_csv(path, parse_dates=['Date'])
    keys = ['Store', 'Dept', 'Date']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate store/department/week records; resolve before analysis.')
    if df[keys + ['Weekly_Sales']].isna().any().any() or not np.isfinite(df.Weekly_Sales).all():
        raise ValueError('Missing or nonfinite sales records; resolve before analysis.')
    return df.sort_values(keys)

def series_for(df, store, dept):
    selected = df.loc[(df.Store == store) & (df.Dept == dept)].set_index('Date').Weekly_Sales.sort_index()
    if selected.empty:
        raise ValueError('No records for this selection.')
    # Missing weeks remain missing: never treat an absent observation as zero sales.
    return selected.reindex(pd.date_range(selected.index.min(), selected.index.max(), freq='7D'))

def forecast(history, horizon=4):
    if history.empty or not np.isfinite(history.iloc[-1]):
        raise ValueError('The latest week is missing; a forecast is unavailable.')
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=7), periods=horizon, freq='7D')
    return pd.DataFrame({'Date': dates, METHODS[0]: [history.iloc[-1]] * horizon,
                         METHODS[1]: [history.get(d - pd.Timedelta(weeks=52), np.nan) for d in dates]})

def backtest(series, origins=12, horizon=4):
    records = []
    # Each origin uses only observations available then; final horizon is fully observed.
    for end in range(max(52, len(series) - horizon - origins + 1), len(series) - horizon + 1):
        history = series.iloc[:end]
        if not np.isfinite(history.iloc[-1]):
            continue
        predictions = forecast(history, horizon)
        for step, row in predictions.iterrows():
            actual = series.get(row.Date, np.nan)
            if not np.isfinite(actual) or any(not np.isfinite(row[m]) for m in METHODS):
                continue  # compare both methods on exactly the same observations
            for method in METHODS:
                records.append({'Origin': history.index[-1], 'Date': row.Date, 'Horizon': step + 1,
                                'Method': method, 'Actual': actual, 'Forecast': row[method]})
    return pd.DataFrame(records, columns=['Origin','Date','Horizon','Method','Actual','Forecast'])

def metrics(results):
    rows = []
    for method, g in results.groupby('Method'):
        error = g.Forecast - g.Actual
        denominator = g.Actual.abs().sum()
        rows.append({'Method': method, 'Evaluated predictions': len(g),
                     'MAE dollars': float(error.abs().mean()), 'Bias dollars': float(error.mean()),
                     'WAPE percent': float(error.abs().sum()/denominator*100) if denominator else None})
    return pd.DataFrame(rows)
