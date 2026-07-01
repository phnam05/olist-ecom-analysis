# 📦 Olist E-Commerce Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phnam05-olist.streamlit.app/)

An interactive Streamlit dashboard for analyzing the Brazilian **Olist e-commerce dataset (2016–2018)**, covering payment behavior, delivery performance, customer retention, RFM segmentation, and churn prediction.

---

## Features

### 📈 EDA
- Monthly order trends with an interactive date range slider
- Year-over-year comparison (2017 vs. 2018)
- Order status breakdown with summary table

### 💳 Payment Behavior
- Payment method distribution by order count, average value, and total revenue
- Credit card installment analysis — order volume and average spend per installment tier

### 🚚 Delivery & Satisfaction
- Late vs. on-time delivery impact on review scores
- Lateness severity bucketing (1–3 days → 15+ days) with average score per bucket
- **Mann-Whitney U test** confirming the score gap is statistically significant (p < 0.001, rank-biserial r ≈ 0.55 — a large effect)

### 🔁 Customer Retention
- One-time vs. returning customer breakdown
- Distribution of days between first and second purchase
- Cumulative return curve with 30 / 90 / 180-day milestones

### 🎯 RFM Segmentation
- Quintile-based Recency, Frequency, and Monetary scoring
- Six named segments are defined (Champions, Loyal, New, Potential Loyalists, At Risk, Lost) — but on this dataset only **three populate** (New, Potential Loyalists, Lost), because ~97% of customers have exactly one order, so the Frequency score can't separate anyone. The app surfaces this honestly rather than hiding it.
- Scatter plot of customer map colored by segment

### 🤖 Churn Prediction
- Binary classification: will a customer make a second purchase?
- Four models compared: Logistic Regression and Gradient Boosting × two imbalance strategies (class weighting and undersampling)
- Metrics: ROC-AUC, Precision, Recall, F1 on the retained class
- Interactive live predictor — adjust order features and get a real-time churn probability estimate

---

## Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| Data manipulation | Pandas, DuckDB |
| Visualization | Altair |
| Machine learning | Scikit-learn |
| Statistics | SciPy |

---

## Project Structure

```
.
├── olist-app.py
├── requirements.txt
├── data/                       # 5 CSVs the app reads (others in the raw dataset are untracked)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_customers_dataset.csv
│   └── olist_order_reviews_dataset.csv
└── README.md
```

---

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
streamlit run olist-app.py
```

> **Data note:** the source CSVs store dates in `DD-MM-YY` format (an Excel-style re-export), so the app parses them with an explicit format. Parsing them naively silently swaps day/month for ~40% of rows and corrupts every time-based metric.

---

## Data Source

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — real-world transactional data covering orders, payments, customers, and reviews from 2016 to 2018.

---

## Key Findings

- Order volume grew ~20× from late 2016 to its November 2017 (Black Friday) peak of ~7,500 orders/month
- Credit cards account for ~75% of transactions and carry the highest average order value
- Late deliveries (~8% of orders) drop average review scores from **4.29 → 2.57** — a ~1.7-point gap, confirmed by a Mann-Whitney U test (large effect)
- ~97% of customers never make a second (delivered) purchase — the median return window for those who do is ~75 days
- Churn is genuinely **low-signal**: the best model reaches only AUC ≈ 0.59, and the strongest predictor is **order value**, not review score (which ranks near the bottom)

---

## Potential Extensions

- Cohort analysis by acquisition month
- Seller-side analytics (top sellers, fulfillment rates)
- Geographic heatmap by state
- Additional ML features (product category, seller region) — though churn here is inherently low-signal, so gains would likely be modest
