from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
from analytics import load_sales, series_for, forecast, backtest, metrics, METHODS

st.set_page_config(page_title='Retail Sales & Forecast Analysis', layout='wide')
st.title('Retail Sales & Forecast Analysis')
st.caption('Sanjay Sarella · Historical Walmart weekly sales · Dollars, not inventory units')
@st.cache_data
def data():
    return load_sales()
try:
    df = data()
except ValueError as e:
    st.error(str(e)); st.stop()
st.info(f'Historical case study: {df.Date.min():%d %b %Y}–{df.Date.max():%d %b %Y}. Forecasts extend from the selected series endpoint, not from today. No stock availability or purchase-order data is present.')
store = st.sidebar.selectbox('Store', sorted(df.Store.unique()))
dept = st.sidebar.selectbox('Department', sorted(df.loc[df.Store == store, 'Dept'].unique()))
s = series_for(df, store, dept)
st.sidebar.caption('Negative sales are retained as recorded. Missing weeks are not filled with zeros.')
performance, evaluation, future = st.tabs(['Sales performance', 'Forecast evaluation', 'Four-week outlook'])
with performance:
    st.subheader(f'Store {store} · Department {dept}')
    a,b,c = st.columns(3)
    a.metric('Recorded sales total', f'${s.sum():,.2f}')
    b.metric('Observed weeks', int(s.notna().sum()))
    c.metric('Missing weeks', int(s.isna().sum()))
    st.line_chart(s.rename('Weekly sales dollars'))
    comparison = pd.DataFrame({'Latest four weeks': s.iloc[-4:].values, 'Previous four weeks': s.iloc[-8:-4].values}) if len(s) >= 8 else pd.DataFrame()
    if len(comparison) and comparison.notna().all().all():
        recent, previous = comparison.sum()
        st.write(f'Latest four-week sales: ${recent:,.2f}; preceding four weeks: ${previous:,.2f}; change: ${recent-previous:,.2f}. This is an observed change, not a causal explanation.')
    st.download_button('Download selected sales', s.rename('Weekly_Sales').to_csv(index_label='Date'), 'selected_sales.csv', 'text/csv')
with evaluation:
    st.subheader('Compare two transparent forecasting baselines')
    st.write('Last observed week repeats the latest available sales. Seasonal baseline uses sales from exactly 52 weeks earlier. Neither uses future observations at a forecast origin.')
    results = backtest(s)
    if results.empty:
        st.warning('Insufficient comparable history for this evaluation.')
    else:
        st.dataframe(metrics(results), hide_index=True)
        st.caption('Up to 12 rolling origins, each predicting horizons 1–4. Overlapping target weeks occur at different horizons; these are not independent observations. Both methods use identical eligible cases. Bias is forecast minus actual; WAPE uses absolute sales in its denominator.')
        st.dataframe(results.groupby(['Method','Horizon']).apply(lambda g: (g.Forecast-g.Actual).abs().mean(), include_groups=False).rename('MAE dollars').reset_index(), hide_index=True)
        st.download_button('Download forecast evidence', results.to_csv(index=False), 'forecast_evaluation.csv', 'text/csv')
with future:
    st.subheader('Four weeks after the historical data cutoff')
    try:
        outlook = forecast(s)
        st.dataframe(outlook, hide_index=True)
        st.caption('Two baseline estimates, not guaranteed outcomes or a prediction interval. Missing seasonal history appears blank. No automated selection based on the evaluation set.')
        st.download_button('Download outlook', outlook.to_csv(index=False), 'historical_outlook.csv', 'text/csv')
    except ValueError as e:
        st.warning(str(e))
with st.expander('Method and boundaries'):
    st.write('Source: repository data/raw/train.csv, Walmart sales forecasting dataset. Record key: store, department, date. Totals reconcile to selected source rows. Evaluation uses only past data at each origin. These outputs support sales review; they cannot establish stockout probability, reorder units, causal holiday effects, or savings.')
