import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import numpy as np
import pandas as pd
import pytest
from analytics import load_sales,series_for,forecast,backtest,metrics,METHODS

def test_known_forecast_and_dates():
    s=pd.Series(np.arange(60.),index=pd.date_range('2020-01-03',periods=60,freq='7D'))
    p=forecast(s)
    assert p[METHODS[0]].tolist()==[59.]*4
    assert p[METHODS[1]].tolist()==[8.,9.,10.,11.]
    assert (p.Date>s.index.max()).all()

def test_no_cross_store_contamination():
    df=pd.DataFrame({'Store':[1,1,2,2],'Dept':[1]*4,'Date':pd.to_datetime(['2020-01-03','2020-01-10']*2),'Weekly_Sales':[100,100,1,1]})
    assert forecast(series_for(df,2,1))[METHODS[0]].tolist()==[1]*4

def test_missing_week_not_zero():
    df=pd.DataFrame({'Store':[1,1],'Dept':[1,1],'Date':pd.to_datetime(['2020-01-03','2020-01-17']),'Weekly_Sales':[100,200]})
    assert pd.isna(series_for(df,1,1).iloc[1])

def test_no_future_leakage():
    s=pd.Series(np.arange(90.),index=pd.date_range('2020-01-03',periods=90,freq='7D'))
    first=backtest(s); changed=s.copy();changed.iloc[-1]=999999
    second=backtest(changed)
    pd.testing.assert_series_equal(first.Forecast,second.Forecast)
    assert (first.Date>first.Origin).all()

def test_zero_denominator_and_insufficient_history():
    s=pd.Series(np.zeros(70),index=pd.date_range('2020-01-03',periods=70,freq='7D'))
    m=metrics(backtest(s))
    assert (m['MAE dollars']==0).all()
    assert m['WAPE percent'].isna().all()
    assert backtest(s.iloc[:10]).empty

def test_duplicate_rejected(tmp_path):
    f=tmp_path/'x.csv'
    f.write_text('Store,Dept,Date,Weekly_Sales\n1,1,2020-01-03,1\n1,1,2020-01-03,2\n')
    with pytest.raises(ValueError):load_sales(f)

def test_real_data_reconciles():
    df=load_sales();s=series_for(df,1,1)
    assert s.sum()==pytest.approx(df.loc[(df.Store==1)&(df.Dept==1),'Weekly_Sales'].sum())
    assert len(backtest(s))==96

def test_streamlit_initial_and_filter():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/Demand_Intelligence.py')).run(timeout=30)
    assert not app.exception
    app.sidebar.selectbox[0].select(2).run(timeout=30)
    assert not app.exception
