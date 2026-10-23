# Customer Engagement & Product Utilization Analytics for Retention Strategy

Behavioural analysis of 10,000 European bank customers: how engagement, product
depth and balance relate to churn.

**Live dashboard:** (add Streamlit link here)

## Pipeline
1. `validate.py` - data validation and cleaning -> `bank_clean.csv`
2. `features.py` - engagement profiles, flags, Relationship Strength Index -> `bank_features.csv`
3. `analysis.py` - KPIs, tables and charts -> `outputs/`
4. `app.py` - Streamlit dashboard

## Key findings
- Inactive members churn at 26.9% vs 14.3% for active members (1.88x).
- Customers with 2 products churn at 7.6%; 1 product 27.7%; 3-4 products 82.7%-100%.
- Half of all high-balance customers are inactive and churn at 30.5%.
- Credit card ownership has almost no retention effect.

## Run locally
    pip install -r requirements.txt
    python validate.py
    python features.py
    python analysis.py
    streamlit run app.py