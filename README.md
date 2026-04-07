# 📊 Sales & Revenue Analytics Dashboard

An interactive dashboard for analyzing retail sales data with built-in forecasting.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)

## Features

- **KPI Cards** — Total Sales, Profit, Orders, Margin, Avg Order Value
- **Monthly Revenue Trend** — Area chart with time-series view
- **Category & Region Breakdown** — Donut chart + grouped bar chart
- **Top Sub-Categories** — Horizontal bar chart ranked by sales
- **Profit Margin Heatmap** — Category × Region cross-analysis
- **6-Month Revenue Forecast** — Prophet model with 90% confidence intervals

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/sales-analytics-dashboard.git
cd sales-analytics-dashboard

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Download the [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) from Kaggle and upload it via the sidebar.

**Expected columns:** `Order Date`, `Sales`, `Profit`, `Category`, `Region`, `Sub-Category`, `Order ID`

## Tech Stack

- **Python** — Data processing
- **Pandas** — EDA & transformations
- **Plotly** — Interactive visualizations
- **Streamlit** — Web dashboard
- **Prophet** — Time-series forecasting
