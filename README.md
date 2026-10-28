# Retail Sales & Forecast Analysis

Historical Walmart sales analysis by Sanjay Sarella. Three outputs: selected sales history, rolling-origin forecast evaluation, and a four-week outlook relative to the historical cutoff. Existing Cloud Run service URL is intended to remain unchanged.

## Run and verify

```sh
python -m pip install -r requirements.txt pytest
python -m streamlit run app/Demand_Intelligence.py
python -m pytest tests -q
```

Source: `data/raw/train.csv`, 421,570 store/department/week records, 2010–2012. Negative recorded sales remain in the data. Missing weeks remain missing, not zero. Duplicate keys or invalid values stop analysis.

Both forecasts are transparent baselines: last observed week and the observation exactly 52 weeks earlier. They predict future dates from each origin without access to later actuals. The evaluation covers up to 12 rolling origins at horizons 1–4, using identical eligible cases for both methods. Overlapping target weeks at different horizons are not independent observations. MAE, signed bias and WAPE are reported. Zero-denominator WAPE is unavailable. Neither method is automatically selected using these evaluation results. No prediction interval is claimed.

## Scope

Dollar sales cannot establish units to reorder, stock availability, lost sales or prevented losses. Those outputs and free-form AI advice are retired from the supported application. No live/current Walmart forecasts are claimed. Historical models, agent modules, notebooks and BI files remain for provenance, but are not used by the supported runtime and must not be cited as current validated results.

## Release status

Candidate implementation, not deployed. Eight automated checks pass, including Streamlit initial rendering and a store change, future-date forecasts, no future leakage, group isolation, missing weeks and source reconciliation. Container/cloud verification is still required. Do not describe this candidate as production-validated.

Deploy a new revision to the existing service `retail-demand-app` in its existing project/region only after verifying its current configuration. Preserve the service, hostname and rollback revision. Do not create a replacement service or redirect the URL.
