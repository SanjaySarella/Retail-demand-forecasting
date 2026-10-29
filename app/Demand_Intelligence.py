from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
from analytics import load_sales, series_for, forecast, backtest, metrics, METHODS

st.set_page_config(page_title='Retail Sales Forecasting | Sanjay Sarella', layout='wide')
st.title('Retail Sales Forecasting')
st.caption('Sanjay Sarella · Data Analytics Portfolio')
st.write('Analyze historical sales, validate forecasting methods, and review a reproducible four-week outlook.')
@st.cache_data
def data():
    return load_sales()
try:
    df = data()
except ValueError as e:
    st.error(str(e)); st.stop()

st.info(f'Dataset coverage: {df.Date.min():%d %b %Y}–{df.Date.max():%d %b %Y}. Select a store and department to review recorded sales, compare forecasting methods, and generate a four-week planning outlook from the historical cutoff.')

st.sidebar.header('Analysis filters')
store = st.sidebar.selectbox('Store location', sorted(df.Store.unique()))
dept = st.sidebar.selectbox('Department', sorted(df.loc[df.Store == store, 'Dept'].unique()))
s = series_for(df, store, dept)
st.sidebar.caption('Recorded negative sales are retained. Missing weeks remain missing and are never replaced with zero.')

performance, evaluation, future = st.tabs(['Historical sales', 'Forecast validation', 'Planning outlook'])
with performance:
    st.subheader(f'Store {store} · Department {dept}')
    a,b,c = st.columns(3)
    a.metric('Total recorded sales', f'${s.sum():,.2f}')
    b.metric('Weeks with sales data', int(s.notna().sum()))
    c.metric('Weeks without sales data', int(s.isna().sum()))
    st.line_chart(s.rename('Weekly sales dollars'))
    comparison = pd.DataFrame({'Latest four weeks': s.iloc[-4:].values, 'Previous four weeks': s.iloc[-8:-4].values}) if len(s) >= 8 else pd.DataFrame()
    if len(comparison) and comparison.notna().all().all():
        recent, previous = comparison.sum()
        st.markdown(f'Latest four-week sales were \\${recent:,.2f}, compared with \\${previous:,.2f} in the preceding four weeks—a recorded change of \\${recent-previous:,.2f}.')
        st.caption('This comparison describes the observed sales movement; it does not assign a cause.')
    st.download_button('Export selected sales data', s.rename('Weekly_Sales').to_csv(index_label='Date'), 'selected_sales.csv', 'text/csv')

with evaluation:
    st.subheader('Forecast method comparison')
    st.write('Compare a last-week baseline with a 52-week seasonal baseline using the same rolling historical test cases. Every forecast uses only information available at its forecast origin.')
    results = backtest(s)
    if results.empty:
        st.warning('This selection does not contain enough comparable history to run the validation.')
    else:
        st.dataframe(metrics(results), hide_index=True)
        st.caption('The evaluation uses up to 12 rolling origins at horizons of one to four weeks. Bias is forecast minus actual; WAPE uses absolute recorded sales in its denominator.')
        st.dataframe(results.groupby(['Method','Horizon']).apply(lambda g: (g.Forecast-g.Actual).abs().mean(), include_groups=False).rename('MAE dollars').reset_index(), hide_index=True)
        st.download_button('Export validation evidence', results.to_csv(index=False), 'forecast_evaluation.csv', 'text/csv')

with future:
    st.subheader('Four-week planning outlook')
    st.write('Review both transparent baseline estimates for the four weeks immediately following the selected series endpoint.')
    try:
        outlook = forecast(s)
        st.dataframe(outlook, hide_index=True)
        st.caption('These are baseline planning estimates, not guaranteed outcomes or prediction intervals. Missing seasonal history remains blank, and the application does not automatically select a preferred method.')
        st.download_button('Export four-week outlook', outlook.to_csv(index=False), 'historical_outlook.csv', 'text/csv')
    except ValueError as e:
        st.warning(str(e))

with st.expander('Data, methods, and decision boundaries'):
    st.write('Source: repository data/raw/train.csv from the Walmart sales forecasting dataset. The record key is store, department, and date. Selected totals reconcile to the underlying source rows, and forecast validation uses only prior observations at each origin.')
    st.write('Use these outputs to review sales patterns and compare simple planning baselines. The dataset does not contain inventory availability, purchase orders, lead times, or intervention outcomes, so the application does not estimate reorder quantities, stockout probability, causal holiday effects, or savings.')
