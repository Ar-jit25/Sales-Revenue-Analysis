import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #22263a 100%);
        border: 1px solid #2d3151;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #f9fafb;
        font-family: 'DM Mono', monospace;
    }
    .metric-delta {
        font-size: 12px;
        color: #34d399;
        margin-top: 4px;
    }
    .section-title {
        font-size: 14px;
        font-weight: 500;
        color: #9ca3af;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 24px;
    }
    .stSelectbox label, .stDateInput label, .stMultiSelect label {
        color: #9ca3af !important;
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='latin-1')
    # Detect date column
    for col in ['Order Date', 'order_date', 'Date', 'date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=False, errors='coerce')
            df.rename(columns={col: 'Order Date'}, inplace=True)
            break
    # Standardise column names (title-case)
    df.columns = [c.strip() for c in df.columns]
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Sales Dashboard")
    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    st.markdown("""
**Expected columns:**
- `Order Date`
- `Sales`
- `Profit`
- `Category`
- `Region`
- `Sub-Category` *(optional)*
- `Customer Name` *(optional)*

Download the [Superstore dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) from Kaggle.
""")

if uploaded is None:
    st.markdown("## 📈 Sales & Revenue Analytics")
    st.info("👈 Upload your Superstore CSV from the sidebar to get started.")
    st.stop()

df = load_data(uploaded)

# ── Date filter ───────────────────────────────────────────────────────────────
if 'Order Date' in df.columns:
    min_d, max_d = df['Order Date'].min(), df['Order Date'].max()
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Filters")
        date_range = st.date_input("Date range", [min_d, max_d])
        if len(date_range) == 2:
            df = df[(df['Order Date'] >= pd.Timestamp(date_range[0])) &
                    (df['Order Date'] <= pd.Timestamp(date_range[1]))]

if 'Region' in df.columns:
    with st.sidebar:
        regions = st.multiselect("Region", df['Region'].unique().tolist(),
                                 default=df['Region'].unique().tolist())
        df = df[df['Region'].isin(regions)]

if 'Category' in df.columns:
    with st.sidebar:
        cats = st.multiselect("Category", df['Category'].unique().tolist(),
                              default=df['Category'].unique().tolist())
        df = df[df['Category'].isin(cats)]


# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("## 📈 Sales & Revenue Analytics")
st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)

total_sales   = df['Sales'].sum()   if 'Sales'  in df.columns else 0
total_profit  = df['Profit'].sum()  if 'Profit' in df.columns else 0
total_orders  = df['Order ID'].nunique() if 'Order ID' in df.columns else len(df)
profit_margin = (total_profit / total_sales * 100) if total_sales else 0
avg_order_val = total_sales / total_orders if total_orders else 0

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "Total Sales",     f"${total_sales:,.0f}",    ""),
    (c2, "Total Profit",    f"${total_profit:,.0f}",   ""),
    (c3, "Total Orders",    f"{total_orders:,}",        ""),
    (c4, "Profit Margin",   f"{profit_margin:.1f}%",    ""),
    (c5, "Avg Order Value", f"${avg_order_val:,.0f}",   ""),
]
for col, label, value, delta in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Row 1: Monthly trend + Category breakdown ─────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="section-title">Monthly Revenue Trend</p>', unsafe_allow_html=True)
    if 'Order Date' in df.columns and 'Sales' in df.columns:
        monthly = (df.set_index('Order Date')['Sales']
                     .resample('M').sum()
                     .reset_index()
                     .rename(columns={'Order Date': 'Month', 'Sales': 'Revenue'}))
        fig = px.area(monthly, x='Month', y='Revenue',
                      color_discrete_sequence=['#6366f1'])
        fig.update_traces(fill='tozeroy', line_width=2,
                          fillcolor='rgba(99,102,241,0.15)')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#9ca3af', margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#1f2937', zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">Sales by Category</p>', unsafe_allow_html=True)
    if 'Category' in df.columns and 'Sales' in df.columns:
        cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
        fig2 = px.pie(cat_sales, values='Sales', names='Category',
                      color_discrete_sequence=['#6366f1', '#8b5cf6', '#a78bfa'],
                      hole=0.55)
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#9ca3af', margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2)
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Region bar + Sub-category ─────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<p class="section-title">Sales & Profit by Region</p>', unsafe_allow_html=True)
    if 'Region' in df.columns:
        reg = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index()
        fig3 = go.Figure()
        fig3.add_bar(x=reg['Region'], y=reg['Sales'],   name='Sales',
                     marker_color='#6366f1')
        fig3.add_bar(x=reg['Region'], y=reg['Profit'],  name='Profit',
                     marker_color='#34d399')
        fig3.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#9ca3af', margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1f2937'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown('<p class="section-title">Top 10 Sub-Categories by Sales</p>', unsafe_allow_html=True)
    if 'Sub-Category' in df.columns and 'Sales' in df.columns:
        sub = (df.groupby('Sub-Category')['Sales'].sum()
                 .nlargest(10).reset_index().sort_values('Sales'))
        fig4 = px.bar(sub, x='Sales', y='Sub-Category', orientation='h',
                      color='Sales', color_continuous_scale='Purples')
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#9ca3af', margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor='#1f2937'),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Profit margin heatmap ──────────────────────────────────────────────
if 'Category' in df.columns and 'Region' in df.columns and 'Profit' in df.columns and 'Sales' in df.columns:
    st.markdown('<p class="section-title">Profit Margin Heatmap — Category × Region</p>', unsafe_allow_html=True)
    pivot = df.groupby(['Category', 'Region']).apply(
        lambda x: (x['Profit'].sum() / x['Sales'].sum() * 100) if x['Sales'].sum() else 0
    ).unstack(fill_value=0)
    fig5 = px.imshow(pivot, text_auto='.1f', aspect='auto',
                     color_continuous_scale='RdYlGn',
                     labels=dict(color='Margin %'))
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#9ca3af', margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── Forecasting ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📅 6-Month Revenue Forecast (Prophet)</p>', unsafe_allow_html=True)

if 'Order Date' in df.columns and 'Sales' in df.columns:
    monthly_prophet = (df.set_index('Order Date')['Sales']
                         .resample('M').sum()
                         .reset_index()
                         .rename(columns={'Order Date': 'ds', 'Sales': 'y'}))

    if len(monthly_prophet) >= 6:
        with st.spinner("Training forecasting model..."):
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, interval_width=0.90)
            m.fit(monthly_prophet)
            future   = m.make_future_dataframe(periods=6, freq='M')
            forecast = m.predict(future)

        fig6 = go.Figure()
        fig6.add_scatter(x=monthly_prophet['ds'], y=monthly_prophet['y'],
                         name='Actual', line=dict(color='#6366f1', width=2))
        fig6.add_scatter(x=forecast['ds'], y=forecast['yhat'],
                         name='Forecast', line=dict(color='#f59e0b', width=2, dash='dot'))
        fig6.add_scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(245,158,11,0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='90% CI'
        )
        fig6.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#9ca3af', margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#1f2937'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Forecast table
        with st.expander("View forecast numbers"):
            future_only = forecast[forecast['ds'] > monthly_prophet['ds'].max()][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            future_only.columns = ['Month', 'Forecast', 'Lower Bound', 'Upper Bound']
            future_only['Month'] = future_only['Month'].dt.strftime('%b %Y')
            for c in ['Forecast', 'Lower Bound', 'Upper Bound']:
                future_only[c] = future_only[c].map('${:,.0f}'.format)
            st.dataframe(future_only, use_container_width=True, hide_index=True)
    else:
        st.warning("Need at least 6 months of data for forecasting.")

st.markdown("---")

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("🗃️ View Raw Data"):
    st.dataframe(df, use_container_width=True)