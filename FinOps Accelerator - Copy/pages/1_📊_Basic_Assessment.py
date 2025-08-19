import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import *

# Set page config
st.set_page_config(
    page_title="Basic Assessment | FinOps Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        height: 140px; /* ensure equal height across all */
        display: flex; /* vertical centering */
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        box-sizing: border-box;
    }
    .kpi-card h2, .kpi-card h3 { margin: 0; }
    .health-score-excellent { color: #28a745; }
    .health-score-good { color: #ffc107; }
    .health-score-attention { color: #dc3545; }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    /* Inline metric cards */
    .metric-row { display: flex; gap: 16px; margin: 8px 0 24px 0; }
    .metric-card-inline { flex: 1; background: #fff; border: 1px solid #e6e9ef; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .metric-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .metric-title { font-weight: 600; color: #1f2937; }
    .metric-sub { color: #6b7280; font-size: 0.9rem; }
    .metric-bar { position: relative; width: 100%; height: 8px; background: #e5effa; border-radius: 6px; overflow: hidden; margin: 8px 0 6px 0; }
    .metric-fill { position: absolute; top: 0; left: 0; height: 100%; background: #1f77b4; border-radius: 6px; }
    .metric-notes { color: #6b7280; font-size: 0.85rem; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# Load CSS
load_css()

# Page title
st.markdown('<h1 class="main-header">📊 Basic Cloud Health Assessment</h1>', unsafe_allow_html=True)

# Generate data
health_metrics = generate_health_metrics()
aws_costs, databricks_costs = generate_cloud_spend_data()
aws_idle, databricks_idle = generate_idle_resources()
untagged_df = generate_untagged_resources()

# Calculate health score
health_score, score_breakdown = calculate_health_score(health_metrics)

# Health Score (top)
st.subheader("🎯 Infrastructure Health Score")
fig_gauge = create_health_score_gauge(health_score)
st.plotly_chart(fig_gauge, use_container_width=True)

# Cloud selection (does not affect Total Cloud Spend chart)
cloud_choice = st.radio(
    "Cloud selection",
    options=["All", "AWS", "Databricks"],
    horizontal=True,
    key="basic_cloud_filter",
    help="Filter KPIs and tables below by cloud. Total Cloud Spend remains unchanged."
)

# Date range selection (filters sections below when a date column exists)
col_sd, col_ed = st.columns(2)
with col_sd:
    start_date = st.date_input("Start date", value=pd.Timestamp.today().normalize().replace(day=1))
with col_ed:
    # default to end of current month
    end_of_month = (pd.Timestamp.today().normalize().to_period('M').to_timestamp('M'))
    end_date = st.date_input("End date", value=end_of_month)

if pd.to_datetime(end_date) < pd.to_datetime(start_date):
    start_date, end_date = end_date, start_date

st.session_state['date_range'] = (pd.to_datetime(start_date), pd.to_datetime(end_date))

# Helpers: detect a date column and filter by date range if present
def _detect_date_col(df: pd.DataFrame):
    for col in ['date', 'Date', 'timestamp', 'Timestamp', 'billing_date', 'BillingDate', 'month', 'Month']:
        if col in df.columns:
            return col
    return None

def _filter_df_by_date(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    col = _detect_date_col(df)
    if not col:
        return df
    ser = pd.to_datetime(df[col], errors='coerce')
    mask = (ser >= pd.to_datetime(start_dt)) & (ser <= pd.to_datetime(end_dt))
    return df[mask].copy()

# Health metrics breakdown
# Data
metrics = [
    {"Metric": "Tagging Compliance", "Score": 97, "Weight": 40, "Impact": "Excellent tagging governance", "Notes": "97% compliance rate", "Icon": "✅"},
    {"Metric": "Idle Usage", "Score": 94, "Weight": 30, "Impact": "12 idle resources found", "Notes": "Optimization needed", "Icon": "⚠"},
    {"Metric": "Connectivity", "Score": 75, "Weight": 20, "Impact": "Azure timeout, GCP not connected", "Notes": "Connection issues", "Icon": "🔌"},
    {"Metric": "Cost Optimization", "Score": 82, "Weight": 10, "Impact": "$3,200 savings pending", "Notes": "Good progress", "Icon": "💰"},
    ]

st.markdown("## 📊 Health Metrics Breakdown")

# Inline rows with blue progress bars
# Row 1: Tagging Compliance, Idle Usage
st.markdown("<div class='metric-row'>" 
            f"<div class='metric-card-inline'>"
            f"  <div class='metric-head'><div class='metric-title'>✅ Tagging Compliance</div><div class='metric-sub'>Weight: {metrics[0]['Weight']}%</div></div>"
            f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[0]['Score']}%'></span></div>"
            f"  <div class='metric-sub'>Score: {metrics[0]['Score']}%</div>"
            f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[0]['Impact']}<br/><em>{metrics[0]['Notes']}</em></div>"
            f"</div>"
            f"<div class='metric-card-inline'>"
            f"  <div class='metric-head'><div class='metric-title'>⚠ Idle Resources Count</div><div class='metric-sub'>Weight: {metrics[1]['Weight']}%</div></div>"
            f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[1]['Score']}%'></span></div>"
            f"  <div class='metric-sub'>Score: {metrics[1]['Score']}%</div>"
            f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[1]['Impact']}<br/><em>{metrics[1]['Notes']}</em></div>"
            f"</div>"
            "</div>", unsafe_allow_html=True)

# # Row 2: Connectivity, Cost Optimization
# st.markdown("<div class='metric-row'>" 
#             f"<div class='metric-card-inline'>"
#             f"  <div class='metric-head'><div class='metric-title'>🔌 Connectivity</div><div class='metric-sub'>Weight: {metrics[2]['Weight']}%</div></div>"
#             f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[2]['Score']}%'></span></div>"
#             f"  <div class='metric-sub'>Score: {metrics[2]['Score']}%</div>"
#             f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[2]['Impact']}<br/><em>{metrics[2]['Notes']}</em></div>"
#             f"</div>"
#             f"<div class='metric-card-inline'>"
#             f"  <div class='metric-head'><div class='metric-title'>💰 Cost Optimization</div><div class='metric-sub'>Weight: {metrics[3]['Weight']}%</div></div>"
#             f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[3]['Score']}%'></span></div>"
#             f"  <div class='metric-sub'>Score: {metrics[3]['Score']}%</div>"
#             f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[3]['Impact']}<br/><em>{metrics[3]['Notes']}</em></div>"
#             f"</div>"
#             "</div>", unsafe_allow_html=True)

st.divider()




# Total Cloud Spend (directly below Health Score)
st.subheader("💰 Total Cloud Spend")

# Single pie chart: AWS vs Databricks totals
total_aws = sum(aws_costs.values())
total_db = sum(databricks_costs.values())
fig_total = go.Figure(data=[
    go.Pie(labels=["AWS", "Databricks"], values=[total_aws, total_db], hole=0.3)
])
fig_total.update_traces(
    texttemplate="$%{value:,.0f}<br>%{percent}",
    textposition="inside",
    hovertemplate="%{percent}<extra></extra>"
)
fig_total.update_layout(
    margin=dict(l=10, r=10, t=40, b=10),
    title_text="AWS vs Databricks Spend",
    width=520,
    height=320
)
st.plotly_chart(fig_total, use_container_width=False)

# Idle Resources KPI Cards
st.markdown("---")
st.subheader("⏰ Idle Resource KPIs")

# Cache base dataframes for idle resources once, and prepare a helper to compute display values per date range
if 'aws_idle_base' not in st.session_state:
    st.session_state['aws_idle_base'] = pd.DataFrame(aws_idle['resources'])
if 'databricks_idle_base' not in st.session_state:
    st.session_state['databricks_idle_base'] = pd.DataFrame(databricks_idle['resources'])

def _prep_idle_display(df_base: pd.DataFrame):
    df_f = _filter_df_by_date(df_base, *st.session_state['date_range'])
    tmp = df_f.copy()
    total_idle_days = 0
    if 'idle_time' in tmp.columns:
        tmp['idle_days'] = tmp['idle_time'].str.extract(r"(\d+)").astype(int)
        total_idle_days = int(tmp['idle_days'].sum())
        tmp = tmp.sort_values('idle_days', ascending=False).drop(columns=['idle_days'])
    count = len(df_f)
    # Try compute total cost from available numeric-like columns
    total_cost = None
    for c in ['cost', 'monthly_cost', 'Cost']:
        if c in df_f.columns:
            vals = pd.to_numeric(df_f[c].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
            total_cost = int(vals.fillna(0).sum())
            break
    return tmp, count, total_idle_days, total_cost

# Compute combined when needed
def get_combined_idle():
    df_all = pd.concat([st.session_state['aws_idle_df'], st.session_state['databricks_idle_df']], ignore_index=True)
    # recompute idle_days for sorting display
    tmp = df_all.copy()
    tmp['idle_days'] = tmp['idle_time'].str.extract(r"(\d+)").astype(int)
    tmp = tmp.sort_values('idle_days', ascending=False).drop(columns=['idle_days'])
    total_count = aws_idle['count'] + databricks_idle['count']
    total_cost = aws_idle['total_cost'] + databricks_idle['total_cost']
    total_idle_days = st.session_state['aws_total_idle_days'] + st.session_state['db_total_idle_days']
    return tmp, total_count, total_cost, total_idle_days

if cloud_choice == "AWS":
    st.markdown("### AWS Idle Resources")
    aws_disp_df, aws_count, aws_idle_days, aws_cost_calc = _prep_idle_display(st.session_state['aws_idle_base'])
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{aws_count}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${(aws_cost_calc if aws_cost_calc is not None else aws_idle['total_cost']):,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{aws_idle_days} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)
elif cloud_choice == "Databricks":
    st.markdown("### Databricks Idle Resources")
    db_disp_df, db_count, db_idle_days, db_cost_calc = _prep_idle_display(st.session_state['databricks_idle_base'])
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{db_count}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${(db_cost_calc if db_cost_calc is not None else databricks_idle['total_cost']):,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{db_idle_days} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)
else:  # All
    aws_disp_df, aws_count, aws_idle_days, aws_cost_calc = _prep_idle_display(st.session_state['aws_idle_base'])
    db_disp_df, db_count, db_idle_days, db_cost_calc = _prep_idle_display(st.session_state['databricks_idle_base'])
    combined_df = pd.concat([aws_disp_df, db_disp_df], ignore_index=True)
    combined_count = aws_count + db_count
    combined_cost = (aws_cost_calc if aws_cost_calc is not None else aws_idle['total_cost']) + (db_cost_calc if db_cost_calc is not None else databricks_idle['total_cost'])
    combined_idle_days = aws_idle_days + db_idle_days
    st.markdown("### All Cloud Idle Resources (AWS + Databricks)")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{combined_count}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${combined_cost:,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{combined_idle_days} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)


