"""Independent forecast/reference reconciliation across all available series."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from analytics import load_sales,series_for,forecast,backtest,METHODS

def run():
 df=load_sales(); failures=[]; evaluated=0; missing=0; seasonal_better=0; naive_better=0; equal=0; cases=0
 for (store,dept),g in df.groupby(['Store','Dept']):
  s=series_for(df,store,dept)
  # Independent source-row total and calendar checks.
  assert np.isclose(s.sum(),sum(g.Weekly_Sales))
  missing+=int(s.isna().sum())
  p=forecast(s)
  for i,row in p.iterrows():
   assert row.Date==s.index[-1]+pd.Timedelta(days=7*(i+1))
   assert row[METHODS[0]]==g.sort_values('Date').Weekly_Sales.iloc[-1]
   source=g.loc[g.Date==row.Date-pd.Timedelta(days=364),'Weekly_Sales']
   expected=float(source.iloc[0]) if len(source) else np.nan
   assert (pd.isna(expected) and pd.isna(row[METHODS[1]])) or expected==row[METHODS[1]]
  # Full evaluation for department 1 across all 45 stores, plus each sparse series.
  if dept!=1 and len(s)>=70:continue
  r=backtest(s)
  if r.empty:continue
  evaluated+=1;cases+=len(r)
  for _,row in r.iterrows():
   expected=s.loc[row.Origin] if row.Method==METHODS[0] else s.loc[row.Date-pd.Timedelta(days=364)]
   assert row.Forecast==expected
   assert row.Actual==s.loc[row.Date]
   assert row.Origin<row.Date
  maes=r.assign(error=(r.Forecast-r.Actual).abs()).groupby('Method').error.mean()
  if maes[METHODS[1]]<maes[METHODS[0]]:seasonal_better+=1
  elif maes[METHODS[1]]>maes[METHODS[0]]:naive_better+=1
  else:equal+=1
 result={'series_reconciled':int(df.groupby(['Store','Dept']).ngroups),'source_rows':len(df),'missing_calendar_weeks':missing,'backtest_series_checked':evaluated,'forecast_cases_checked':cases,'seasonal_lower_mae_series':seasonal_better,'last_week_lower_mae_series':naive_better,'equal_mae_series':equal,'scope':'All series totals and four-week forecasts; full rolling evaluation for department 1 in every store and series shorter than 70 weeks. Historical evaluation, not production validation.'}
 print(json.dumps(result,indent=2));Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run()
