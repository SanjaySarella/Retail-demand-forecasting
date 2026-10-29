# Retail Sales Forecasting

A recruiter-facing analytics case study that compares transparent forecasting baselines for weekly store-department sales. The application answers a bounded question: **which simple method produced lower historical forecast error for the selected series, and what four-week outlook does each method produce from the dataset cutoff?**

## Dataset

The repository includes the Walmart weekly sales training data used by the application. The analytical key is `Store`, `Dept`, and `Date`. The application retains recorded negative sales, leaves absent weeks missing, rejects duplicate keys, and never converts missing weeks to zero sales.

## Workflow

1. Filter one store-department series.
2. Reconcile selected totals to the source rows.
3. Backtest a last-observed-week baseline and a 52-week seasonal baseline on identical rolling origins.
4. Compare MAE, bias, WAPE, and error by forecast horizon.
5. Recommend a baseline only when it has lower MAE and WAPE on the shared evaluation cases.
6. Generate a four-week forecast outlook from the historical cutoff.

## Decision supported

The project supports selection between two transparent baselines for a chosen historical series. It does not estimate inventory requirements, reorder quantities, stockouts, causal holiday effects, or savings because the dataset does not contain inventory, purchase-order, lead-time, or intervention data.

## Validation

- Forecast origins use only observations available at that time.
- Both methods are compared on the same actual observations.
- Source totals and evaluation outputs are independently reconciled in `validation/acceptance.py`.
- Automated tests cover leakage, missing-week handling, store separation, duplicate rejection, metrics, and decision-summary logic.

## Tools

Python, pandas, NumPy, Streamlit, pytest, Docker, Google Cloud Run. A Power BI artifact is retained in `docs/`, and the portable SQL workflow is documented in `sql/analysis.sql`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/Demand_Intelligence.py
pytest -q
python validation/acceptance.py
```

## Live application

https://retail-demand-app-646m5mi6fq-uc.a.run.app/
