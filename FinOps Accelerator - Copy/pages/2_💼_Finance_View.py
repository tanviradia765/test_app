import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_css
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(
    page_title="Finance View | FinOps Dashboard",
    page_icon="💼",
    layout="wide"
)

# Load CSS
load_css()

# Page title
st.markdown('<h1 class="main-header">💼 Finance View</h1>', unsafe_allow_html=True)

# KPI card CSS
st.markdown("""
<style>
.kpi-wrap { display: flex; flex-direction: column; gap: 10px; }
.kpi-card-fin { background: #ffffff; border: 1px solid #e6e9ef; border-radius: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); padding: 16px; height: 140px; display: flex; flex-direction: column; justify-content: space-between; }
.kpi-head { display: flex; align-items: center; justify-content: space-between; }
.kpi-title { font-weight: 600; color: #111827; }
.kpi-badge { padding: 4px 10px; border-radius: 9999px; font-size: 12px; color: #fff; }
.kpi-value { font-size: 26px; font-weight: 700; color: #111827; }
.kpi-context { color: #6b7280; font-size: 13px; }
.badge-green { background: #10b981; }
.badge-orange { background: #f59e0b; }
.badge-red { background: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Cloud selection (for filtering sections below when wired)
cloud_choice_fin = st.radio(
    "Cloud selection",
    options=["All", "AWS", "Databricks"],
    horizontal=True,
    key="finance_cloud_filter",
    help="Filter Finance view sections by cloud (when applicable)."
)

# Date range selection (for filtering sections below when a date column exists)
col_sd_fin, col_ed_fin = st.columns(2)
with col_sd_fin:
    start_date_fin = st.date_input("Start date", value=pd.Timestamp.today().normalize().replace(day=1), key="finance_start_date")
with col_ed_fin:
    end_of_month_fin = (pd.Timestamp.today().normalize().to_period('M').to_timestamp('M'))
    end_date_fin = st.date_input("End date", value=end_of_month_fin, key="finance_end_date")

if pd.to_datetime(end_date_fin) < pd.to_datetime(start_date_fin):
    start_date_fin, end_date_fin = end_date_fin, start_date_fin

st.session_state['finance_date_range'] = (pd.to_datetime(start_date_fin), pd.to_datetime(end_date_fin))

# Helpers: detect a date column and filter by date range if present (Finance view)
def _detect_date_col_fin(df: pd.DataFrame):
    for col in ['date', 'Date', 'timestamp', 'Timestamp', 'billing_date', 'BillingDate', 'month', 'Month']:
        if col in df.columns:
            return col
    return None

def _filter_df_by_date_fin(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    col = _detect_date_col_fin(df)
    if not col:
        return df
    ser = pd.to_datetime(df[col], errors='coerce')
    mask = (ser >= pd.to_datetime(start_dt)) & (ser <= pd.to_datetime(end_dt))
    return df[mask].copy()

# Sample data generation
@st.cache_data
def generate_finance_data():
    # Monthly spend data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    aws_spend = [120000, 118000, 125000, 130000, 128000, 135000, 140000, 138000, 142000, 145000, 148000, 150000]
    databricks_spend = [90000, 92000, 95000, 98000, 100000, 105000, 108000, 110000, 112000, 115000, 118000, 120000]
    
    # Budget data
    budget = [150000] * 12
    
    # Cost by service
    services = ['Compute', 'Storage', 'Database', 'Networking', 'Analytics', 'AI/ML', 'Security']
    aws_costs = [45000, 12000, 18000, 8000, 15000, 22000, 6000]
    db_costs = [35000, 8000, 0, 5000, 18000, 25000, 7000]
    
    return {
        'months': months,
        'aws_spend': aws_spend,
        'databricks_spend': databricks_spend,
        'budget': budget,
        'services': services,
        'aws_costs': aws_costs,
        'db_costs': db_costs
    }

# Generate data
finance_data = generate_finance_data()

# Build monthly DataFrame with dates (placeholder year = current year)
current_year = pd.Timestamp.today().year
month_to_num = {m:i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], start=1)}
df_months = pd.DataFrame({
    'MonthName': finance_data['months'],
    'MonthNum': [month_to_num[m] for m in finance_data['months']],
    'AWS': finance_data['aws_spend'],
    'Databricks': finance_data['databricks_spend'],
    'Budget': finance_data['budget']
})
df_months['Date'] = pd.to_datetime({
    'year': [current_year]*len(df_months),
    'month': df_months['MonthNum'],
    'day': 1
})

# Apply date filter (by month starts intersecting selected range)
start_dt, end_dt = st.session_state.get('finance_date_range', (pd.Timestamp(current_year, 1, 1), pd.Timestamp(current_year, 12, 31)))
mask = (df_months['Date'] >= pd.to_datetime(start_dt)) & (df_months['Date'] <= pd.to_datetime(end_dt))
df_f = df_months[mask].copy()
if df_f.empty:
    df_f = df_months.copy()  # fallback to all months if filter yields none

# Cloud filter
cloud = st.session_state.get('finance_cloud_filter', 'All')
if cloud == 'AWS':
    spend_series = df_f['AWS']
    cloud_label = 'AWS'
elif cloud == 'Databricks':
    spend_series = df_f['Databricks']
    cloud_label = 'Databricks'
else:
    spend_series = df_f['AWS'] + df_f['Databricks']
    cloud_label = 'All Clouds'

# Aggregates
budget_total = float(df_f['Budget'].sum())
actual_spend = float(spend_series.sum())
days_in_range = int((pd.to_datetime(end_dt) - pd.to_datetime(start_dt)).days) + 1
days_in_range = max(days_in_range, 1)
avg_daily_spend = actual_spend / days_in_range

# If single month selected, compute forecast for the month; else extrapolate for the whole range
unique_months = df_f['Date'].dt.to_period('M').unique()
if len(unique_months) == 1:
    period = unique_months[0]
    month_start = period.to_timestamp(how='start')
    month_end = period.to_timestamp(how='end')
    # Days counted within selected range but capped to the month
    from_dt = max(pd.to_datetime(start_dt), month_start)
    to_dt = min(pd.to_datetime(end_dt), month_end)
    days_elapsed = (to_dt - from_dt).days + 1
    days_in_month = (month_end - month_start).days + 1
    # Use current avg_daily across the selected portion
    forecast_spend = avg_daily_spend * days_in_month
    budget_context = f"Allocated Budget for {month_start.strftime('%B %Y')} – {cloud_label}"
else:
    # Multi-month: simple extrapolation across the selected range
    forecast_spend = avg_daily_spend * days_in_range
    budget_context = f"Allocated Budget for selected period – {cloud_label}"

# Status helpers
def status_badge(ratio: float):
    if ratio < 0.8:
        return 'badge-green', 'Within Budget'
    if ratio <= 1.0:
        return 'badge-orange', 'Near Budget'
    return 'badge-red', 'Over Budget'

def render_kpi(title: str, value_str: str, context: str, ratio: float, extra: str = ""):
    badge_cls, badge_text = status_badge(ratio)
    st.markdown(f"""
    <div class='kpi-card-fin'>
      <div class='kpi-head'>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-badge {badge_cls}'>{badge_text}</div>
      </div>
      <div class='kpi-value'>{value_str}</div>
      <div class='kpi-context'>{context}{(' · ' + extra) if extra else ''}</div>
    </div>
    """, unsafe_allow_html=True)

# Compute ratios
spend_vs_budget = (actual_spend / budget_total) if budget_total > 0 else 0
forecast_vs_budget = (forecast_spend / budget_total) if budget_total > 0 else 0
daily_budget = (budget_total / days_in_range) if days_in_range > 0 else 0
daily_vs_budget = (avg_daily_spend / daily_budget) if daily_budget > 0 else 0

# 2x2 KPI layout
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    render_kpi(
        title="Budget",
        value_str=f"${budget_total:,.0f}",
        context=budget_context,
        ratio=spend_vs_budget,
        extra=f"Remaining: ${max(budget_total-actual_spend, 0):,.0f}"
    )
with row1_col2:
    render_kpi(
        title="Spend",
        value_str=f"${actual_spend:,.0f}",
        context="Total Cloud Spend for the selected date range",
        ratio=spend_vs_budget,
        extra=f"{spend_vs_budget*100:,.1f}% of Budget"
    )

# Spacer between top and bottom KPI rows
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    render_kpi(
        title="Forecast Spend",
        value_str=f"${forecast_spend:,.0f}",
        context="Forecasted Spend for the selected period",
        ratio=forecast_vs_budget,
        extra=f"{forecast_vs_budget*100:,.1f}% of Budget"
    )
with row2_col2:
    render_kpi(
        title="Average Daily Spend",
        value_str=f"${avg_daily_spend:,.0f}",
        context="Average Daily Spend for selected period",
        ratio=daily_vs_budget,
        extra=f"Daily Budget: ${daily_budget:,.0f}"
    )

# Spacer before Spend vs Budget Trend heading
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Spend vs Budget Trend
st.subheader("📈 Spend vs Budget Trend")

# View selector
view_choice = st.selectbox(
    "View granularity",
    options=["Monthly", "Weekly"],
    index=0,
    key="finance_trend_view",
    help="Switch between Monthly and Weekly view."
)

# Build daily baseline from monthly values for selected cloud
today = pd.Timestamp.today().normalize()
date_idx = pd.date_range(start=start_dt, end=end_dt, freq='D')

# Choose cloud series
if cloud == 'AWS':
    df_cloud = df_months[['Date', 'AWS', 'Budget']].rename(columns={'AWS': 'Spend'})
elif cloud == 'Databricks':
    df_cloud = df_months[['Date', 'Databricks', 'Budget']].rename(columns={'Databricks': 'Spend'})
else:
    # All = AWS + Databricks
    temp = df_months[['Date', 'AWS', 'Databricks', 'Budget']].copy()
    temp['Spend'] = temp['AWS'] + temp['Databricks']
    df_cloud = temp[['Date', 'Spend', 'Budget']]

# Map month period to spend and budget
df_cloud['Period'] = df_cloud['Date'].dt.to_period('M')
spend_by_period = df_cloud.set_index('Period')['Spend']
budget_by_period = df_cloud.set_index('Period')['Budget']

# Daily frame
daily = pd.DataFrame({'Date': date_idx})
daily['MPer'] = daily['Date'].dt.to_period('M')
daily['DaysInMonth'] = daily['MPer'].dt.days_in_month
daily['SpendMonth'] = daily['MPer'].map(spend_by_period)
daily['BudgetMonth'] = daily['MPer'].map(budget_by_period)

# Handle potential missing months by forward/backfill if needed
daily['SpendMonth'] = daily['SpendMonth'].ffill().bfill()
daily['BudgetMonth'] = daily['BudgetMonth'].ffill().bfill()

# Baseline daily values derived from monthly totals
daily['ActualDailyBase'] = daily['SpendMonth'] / daily['DaysInMonth']
daily['BudgetDaily'] = daily['BudgetMonth'] / daily['DaysInMonth']

# Actual portion = up to today; Projection = after today
past_mask = daily['Date'] <= today
actual_spend_past = float(daily.loc[past_mask, 'ActualDailyBase'].sum())
days_past = int(past_mask.sum())
if days_past > 0:
    avg_daily_past = actual_spend_past / days_past
else:
    # Fallback: average of baseline over selected period
    avg_daily_past = float(daily['ActualDailyBase'].mean()) if not daily.empty else 0.0

daily['ActualDaily'] = daily['ActualDailyBase'].where(past_mask, 0.0)
daily['ProjDaily'] = 0.0
daily.loc[~past_mask, 'ProjDaily'] = avg_daily_past

# Aggregate by selected granularity
if view_choice == 'Monthly':
    daily['Bucket'] = daily['Date'].dt.to_period('M').dt.to_timestamp()
    x_title = 'Month'
else:
    # Weeks starting Monday
    daily['Bucket'] = daily['Date'].dt.to_period('W-MON').dt.start_time
    x_title = 'Week'

grp = daily.groupby('Bucket', as_index=False).agg(
    SpendActual=('ActualDaily', 'sum'),
    Budget=('BudgetDaily', 'sum'),
    Forecast=('ProjDaily', 'sum')
)

# Hide forecast for fully past buckets
if view_choice == 'Monthly':
    bucket_end = grp['Bucket'].dt.to_period('M').dt.end_time
else:
    bucket_end = grp['Bucket'] + pd.to_timedelta(6, unit='D')
grp.loc[bucket_end < today, 'Forecast'] = None

# Plot
fig_spend_budget = go.Figure()
fig_spend_budget.add_trace(go.Scatter(
    x=np.array(grp['Bucket'].dt.to_pydatetime()), y=grp['SpendActual'], mode='lines+markers', name='Spend', line=dict(color='#1f77b4')
))
fig_spend_budget.add_trace(go.Scatter(
    x=np.array(grp['Bucket'].dt.to_pydatetime()), y=grp['Budget'], mode='lines', name='Budget', line=dict(color='#2ca02c')
))
fig_spend_budget.add_trace(go.Scatter(
    x=np.array(grp['Bucket'].dt.to_pydatetime()), y=grp['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='#ff7f0e', dash='dash')
))

fig_spend_budget.update_layout(
    title=f"Spend vs Budget Trend ({view_choice}) — {cloud_label}",
    xaxis_title=x_title,
    yaxis_title='Amount ($)',
    showlegend=True,
    height=500
)

st.plotly_chart(fig_spend_budget, use_container_width=True)

# Cost by Service
st.subheader("🔍 Cost Breakdown by Service")

# Month and Year selectors
sel_cols = st.columns(2)
with sel_cols[0]:
    month_options = finance_data['months']
    sel_month = st.selectbox(
        "Select month",
        options=month_options,
        index=min(pd.Timestamp.today().month - 1, len(month_options) - 1),
        key="finance_service_month"
    )
with sel_cols[1]:
    year_options = list(range(1950, 2051))
    default_year_index = year_options.index(pd.Timestamp.today().year) if pd.Timestamp.today().year in year_options else year_options.index(current_year)
    sel_year = st.selectbox(
        "Select year",
        options=year_options,
        index=default_year_index,
        key="finance_service_year"
    )

# Compute month index and representative spends
mi = month_options.index(sel_month)
aws_month_spend = finance_data['aws_spend'][mi]
db_month_spend = finance_data['databricks_spend'][mi]

# Respect Date Range filter: only show if selected month/year intersects range
sel_month_num = month_to_num[sel_month]
sel_month_start = pd.Timestamp(sel_year, sel_month_num, 1)
sel_month_end = sel_month_start.to_period('M').end_time
in_range = not (sel_month_end < pd.to_datetime(start_dt) or sel_month_start > pd.to_datetime(end_dt))

if not in_range:
    st.info("Selected month/year is outside the chosen Date Range. Adjust the Date Range or Month/Year to view the breakdown.")
else:
    # Distributions (percentages) by service/SKU
    services = finance_data['services']
    aws_costs = pd.Series(finance_data['aws_costs'], index=services)
    db_costs = pd.Series(finance_data['db_costs'], index=services)
    aws_pct = aws_costs / aws_costs.sum() if aws_costs.sum() else aws_costs
    db_pct = db_costs / db_costs.sum() if db_costs.sum() else db_costs

    # Build values by cloud selection
    if cloud == 'AWS':
        pie_values = (aws_pct * aws_month_spend).values
        pie_title = f"AWS Cost Distribution — {sel_month} {sel_year}"
    elif cloud == 'Databricks':
        pie_values = (db_pct * db_month_spend).values
        pie_title = f"Databricks Cost Distribution — {sel_month} {sel_year}"
    else:
        combined = (aws_pct * aws_month_spend) + (db_pct * db_month_spend)
        pie_values = combined.values
        pie_title = f"All Clouds Cost Distribution — {sel_month} {sel_year}"

    fig_service = px.pie(
        names=services,
        values=pie_values,
        title=pie_title,
        hole=0.4
    )
    fig_service.update_traces(textposition='inside')
    fig_service.update_layout(showlegend=True)
    st.plotly_chart(fig_service, use_container_width=True)

# Cost Optimization Opportunities
st.subheader("🏷️ Spend Breakdown by Team/Project")

# Monthly selectors for Team/Project breakdown (separate from service section)
proj_cols = st.columns(2)
with proj_cols[0]:
    proj_month = st.selectbox(
        "Select month",
        options=finance_data['months'],
        index=min(pd.Timestamp.today().month - 1, len(finance_data['months']) - 1),
        key="finance_proj_month"
    )
with proj_cols[1]:
    proj_year_options = list(range(1950, 2051))
    proj_year_default = pd.Timestamp.today().year if pd.Timestamp.today().year in proj_year_options else current_year
    proj_year = st.selectbox(
        "Select year",
        options=proj_year_options,
        index=proj_year_options.index(proj_year_default),
        key="finance_proj_year"
    )

# Synthetic tag-based allocation by team/project (percent split)
teams = [
    'Development',
    'Testing',
    'DevOps',
    'Security',
    'Data Engineering',
    'Analytics',
    'IT Operations',
    'Platform'
]
# Ensure allocations sum to 1.0
aws_alloc = pd.Series([0.28, 0.12, 0.18, 0.10, 0.12, 0.08, 0.07, 0.05], index=teams)
db_alloc = pd.Series([0.22, 0.10, 0.12, 0.08, 0.20, 0.18, 0.06, 0.04], index=teams)

# Determine selected month spend for each cloud
proj_mi = finance_data['months'].index(proj_month)
proj_aws_spend = finance_data['aws_spend'][proj_mi]
proj_db_spend = finance_data['databricks_spend'][proj_mi]

# Respect Date Range filter for the selected month/year
proj_month_num = month_to_num[proj_month]
proj_start = pd.Timestamp(proj_year, proj_month_num, 1)
proj_end = proj_start.to_period('M').end_time
proj_in_range = not (proj_end < pd.to_datetime(start_dt) or proj_start > pd.to_datetime(end_dt))

if not proj_in_range:
    st.info("Selected month/year is outside the chosen Date Range. Adjust the Date Range or Month/Year to view the breakdown.")
else:
    if cloud == 'AWS':
        values = (aws_alloc * proj_aws_spend).values
        proj_title = f"AWS Spend by Team/Project — {proj_month} {proj_year}"
    elif cloud == 'Databricks':
        values = (db_alloc * proj_db_spend).values
        proj_title = f"Databricks Spend by Team/Project — {proj_month} {proj_year}"
    else:
        combined_values = (aws_alloc * proj_aws_spend) + (db_alloc * proj_db_spend)
        values = combined_values.values
        proj_title = f"All Clouds Spend by Team/Project — {proj_month} {proj_year}"

    fig_proj = px.bar(
        x=teams,
        y=values,
        labels={'x': 'Team/Project', 'y': 'Spend ($)'},
        title=proj_title
    )
    fig_proj.update_layout(xaxis_title='Team/Project', yaxis_title='Spend ($)', height=480)
    st.plotly_chart(fig_proj, use_container_width=True)

# Tagging Compliance KPI
st.markdown("---")

# Month/Year selectors for Tagging KPI (cloud uses top-level toggle)
tc_cols = st.columns(2)
with tc_cols[0]:
    tc_month = st.selectbox(
        "Select month",
        options=finance_data['months'],
        index=min(pd.Timestamp.today().month - 1, len(finance_data['months']) - 1),
        key="finance_tag_month"
    )
with tc_cols[1]:
    tc_year_options = list(range(1950, 2051))
    tc_year_default = pd.Timestamp.today().year if pd.Timestamp.today().year in tc_year_options else current_year
    tc_year = st.selectbox(
        "Select year",
        options=tc_year_options,
        index=tc_year_options.index(tc_year_default),
        key="finance_tag_year"
    )
# Use existing top-level cloud toggle for provider selection
cloud_toggle_fin = st.session_state.get('finance_cloud_filter', 'All')
if cloud_toggle_fin == 'AWS':
    sel_providers = ["AWS"]
elif cloud_toggle_fin == 'Databricks':
    sel_providers = ["Databricks"]
else:
    sel_providers = ["AWS", "Databricks"]

@st.cache_data
def generate_tagging_kpi_data(year: int):
    months = list(range(1, 13))
    providers = ["AWS", "Azure", "GCP", "Databricks"]
    rows = []
    rng = np.random.default_rng(7)
    base_tc = {"AWS": 88, "Azure": 85, "GCP": 90, "Databricks": 82}
    base_ip = {"AWS": 14, "Azure": 16, "GCP": 12, "Databricks": 18}  # penalty in % (lower is better)
    for m in months:
        for p in providers:
            noise_tc = rng.normal(0, 2)
            noise_ip = rng.normal(0, 1.5)
            tc = float(np.clip(base_tc[p] + noise_tc + (m-6.5)*0.15, 70, 99))
            ip = float(np.clip(base_ip[p] + noise_ip + (6.5-m)*0.12, 5, 30))
            rows.append({
                'Date': pd.Timestamp(year, m, 1),
                'Provider': p,
                'TaggingCompliance': tc,
                'IdlePenalty': ip
            })
    return pd.DataFrame(rows)

# Build month range and in-range check (optional filter)
tc_month_num = month_to_num[tc_month]
tc_start = pd.Timestamp(tc_year, tc_month_num, 1)
tc_end = tc_start.to_period('M').end_time
tc_in_range = not (tc_end < pd.to_datetime(start_dt) or tc_start > pd.to_datetime(end_dt))

if not tc_in_range:
    st.info("Selected month/year is outside the chosen Date Range. Adjust filters to align if desired.")

df_tc = generate_tagging_kpi_data(tc_year)
cur_df = df_tc[(df_tc['Date'] == tc_start) & (df_tc['Provider'].isin(sel_providers))]
prev_month = (tc_start - pd.offsets.MonthBegin(1))
prev_df = df_tc[(df_tc['Date'] == prev_month) & (df_tc['Provider'].isin(sel_providers))]

# Aggregate across selected providers
cur_tc = float(cur_df['TaggingCompliance'].mean()) if not cur_df.empty else np.nan
cur_ip = float(cur_df['IdlePenalty'].mean()) if not cur_df.empty else np.nan
prev_tc = float(prev_df['TaggingCompliance'].mean()) if not prev_df.empty else np.nan
prev_ip = float(prev_df['IdlePenalty'].mean()) if not prev_df.empty else np.nan

weighted = (cur_tc * 0.4) + (cur_ip * 0.6) if not (np.isnan(cur_tc) or np.isnan(cur_ip)) else np.nan

# Trends
delta_tc = None if np.isnan(prev_tc) or np.isnan(cur_tc) else (cur_tc - prev_tc)
delta_ip = None if np.isnan(prev_ip) or np.isnan(cur_ip) else (cur_ip - prev_ip)

# Arrows: Tagging higher is better; Idle Penalty lower is better
def arrow_html(val: float, better_is_up: bool):
    if val is None:
        return ""
    sym = "▲" if (val >= 0 if better_is_up else val < 0) else "▼"
    good = (val >= 0) if better_is_up else (val < 0)
    color = "#10b981" if good else "#ef4444"
    return f"<span style='color:{color}; font-weight:600; padding-left:4px;'>{sym} {abs(val):.1f} pts</span>"

tag_trend = arrow_html(delta_tc, better_is_up=True)
pen_trend = arrow_html(delta_ip, better_is_up=False)

# Badge color by weighted score
if not np.isnan(weighted):
    if weighted >= 90:
        badge_cls, badge_text = 'badge-green', 'Excellent'
    elif weighted >= 70:
        badge_cls, badge_text = 'badge-orange', 'Fair'
    else:
        badge_cls, badge_text = 'badge-red', 'Poor'
else:
    badge_cls, badge_text = 'badge-orange', 'N/A'

# Tooltip content
tooltip = (
    f"Tagging Compliance: {cur_tc:.1f}%\n"
    f"Idle Resource Penalty: {cur_ip:.1f}%\n"
    f"Formula: (Tagging x 0.4) + (Penalty x 0.6) = {weighted:.1f}%"
    if not (np.isnan(cur_tc) or np.isnan(cur_ip) or np.isnan(weighted))
    else "Insufficient data"
)

# Render KPI card
weighted_str = f"{weighted:.1f}%" if not np.isnan(weighted) else "N/A"
tagging_str = f"{cur_tc:.1f}%" if not np.isnan(cur_tc) else "N/A"
penalty_str = f"{cur_ip:.1f}%" if not np.isnan(cur_ip) else "N/A"
st.markdown(f"""
<div class='kpi-card-fin' title="{tooltip}">
  <div class='kpi-head'>
    <div class='kpi-title'>Tagging Compliance %</div>
    <div class='kpi-badge {badge_cls}'>{badge_text}</div>
  </div>
  <div class='kpi-value'>{weighted_str}</div>
  <div class='kpi-context'>Providers: {', '.join(sel_providers)} · {tc_start.strftime('%b %Y')}</div>
  <div class='kpi-context'>Tagging: {tagging_str} {tag_trend} · Idle Penalty: {penalty_str} {pen_trend}</div>
</div>
""", unsafe_allow_html=True)
