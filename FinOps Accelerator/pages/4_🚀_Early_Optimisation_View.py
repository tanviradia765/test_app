import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_css, generate_idle_resources, generate_untagged_resources, generate_cloud_spend_data
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Early Optimisation | FinOps Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Load CSS
load_css()

# Page title
st.markdown('<h1 class="main-header">🚀 Early Optimisation View</h1>', unsafe_allow_html=True)

# Minimal CSS for KPI cards (matching Basic Assessment)
st.markdown("""
<style>
  .kpi-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; box-sizing: border-box; }
  .kpi-card h3 { margin: 0; }
</style>
""", unsafe_allow_html=True)

# Cloud selection (for filtering sections below when wired)
cloud_choice_opt = st.radio(
    "Cloud selection",
    options=["All", "AWS", "Databricks"],
    horizontal=True,
    key="opt_cloud_filter",
    help="Filter Early Optimisation sections by cloud (when applicable)."
)

# Date range selection (for filtering sections below when a date column exists)
col_sd_opt, col_ed_opt = st.columns(2)
with col_sd_opt:
    start_date_opt = st.date_input("Start date", value=pd.Timestamp.today().normalize().replace(day=1), key="opt_start_date")
with col_ed_opt:
    end_of_month_opt = (pd.Timestamp.today().normalize().to_period('M').to_timestamp('M'))
    end_date_opt = st.date_input("End date", value=end_of_month_opt, key="opt_end_date")

if pd.to_datetime(end_date_opt) < pd.to_datetime(start_date_opt):
    start_date_opt, end_date_opt = end_date_opt, start_date_opt

# Store full-day datetime bounds to avoid 00:00:00-only ranges
_start_dt_full = pd.to_datetime(start_date_opt).replace(hour=0, minute=0, second=0, microsecond=0)
_end_dt_full = pd.to_datetime(end_date_opt).replace(hour=23, minute=59, second=59, microsecond=999999)
st.session_state['opt_date_range'] = (_start_dt_full, _end_dt_full)

# Helpers: detect a date column and filter by date range if present (Early Optimisation view)
def _detect_date_col_opt(df: pd.DataFrame):
    for col in ['date', 'Date', 'timestamp', 'Timestamp', 'billing_date', 'BillingDate', 'month', 'Month', 'Start', 'End']:
        if col in df.columns:
            return col
    return None

def _filter_df_by_date_opt(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    col = _detect_date_col_opt(df)
    if not col:
        return df
    ser = pd.to_datetime(df[col], errors='coerce')
    mask = (ser >= pd.to_datetime(start_dt)) & (ser <= pd.to_datetime(end_dt))
    return df[mask].copy()

# Idle Resources KPI Cards (ported from Basic Assessment)
st.markdown("---")
st.subheader("⏰ Idle Resource KPIs")

# Generate or retrieve idle resources data
aws_idle_opt, databricks_idle_opt = generate_idle_resources()

# Cache base dataframes for idle resources once
if 'opt_aws_idle_base' not in st.session_state:
    st.session_state['opt_aws_idle_base'] = pd.DataFrame(aws_idle_opt['resources'])
if 'opt_databricks_idle_base' not in st.session_state:
    st.session_state['opt_databricks_idle_base'] = pd.DataFrame(databricks_idle_opt['resources'])

def _prep_idle_display_opt(df_base: pd.DataFrame):
    df_f = _filter_df_by_date_opt(df_base, *st.session_state['opt_date_range'])
    tmp = df_f.copy()
    total_idle_days = 0
    if 'idle_time' in tmp.columns:
        tmp['idle_days'] = tmp['idle_time'].str.extract(r"(\d+)").astype(int)
        total_idle_days = int(tmp['idle_days'].sum())
        tmp = tmp.sort_values('idle_days', ascending=False).drop(columns=['idle_days'])
    count = len(df_f)
    total_cost = None
    for c in ['cost', 'monthly_cost', 'Cost']:
        if c in df_f.columns:
            vals = pd.to_numeric(df_f[c].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
            total_cost = int(vals.fillna(0).sum())
            break
    return tmp, count, total_idle_days, total_cost

if cloud_choice_opt == "AWS":
    st.markdown("### AWS Idle Resources")
    disp_df, count_val, idle_days_val, cost_calc = _prep_idle_display_opt(st.session_state['opt_aws_idle_base'])
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{count_val}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${((cost_calc if cost_calc is not None else aws_idle_opt['total_cost'])):,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{idle_days_val} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)
    st.table(disp_df)
elif cloud_choice_opt == "Databricks":
    st.markdown("### Databricks Idle Resources")
    disp_df, count_val, idle_days_val, cost_calc = _prep_idle_display_opt(st.session_state['opt_databricks_idle_base'])
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{count_val}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${((cost_calc if cost_calc is not None else databricks_idle_opt['total_cost'])):,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{idle_days_val} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)
    st.table(disp_df)
else:
    disp_df_aws, count_aws, idle_days_aws, cost_aws = _prep_idle_display_opt(st.session_state['opt_aws_idle_base'])
    disp_df_db, count_db, idle_days_db, cost_db = _prep_idle_display_opt(st.session_state['opt_databricks_idle_base'])
    combined_df = pd.concat([disp_df_aws, disp_df_db], ignore_index=True)
    combined_count = count_aws + count_db
    combined_cost = (cost_aws if cost_aws is not None else aws_idle_opt['total_cost']) + (cost_db if cost_db is not None else databricks_idle_opt['total_cost'])
    combined_idle_days = idle_days_aws + idle_days_db
    st.markdown("### All Cloud Idle Resources")
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
    st.table(combined_df)

# Untagged Resources (moved from Basic Assessment)
st.markdown("---")
st.subheader("🏷️ Untagged Resources")
st.markdown("Resources missing one or more mandatory tags as per organization's tagging policy:")

# Load data once and cache for this view
if 'opt_untagged_df' not in st.session_state:
    st.session_state['opt_untagged_df'] = generate_untagged_resources()

# Enrich with Date_of_creation once (deterministic synthetic dates)
if 'Date_of_creation' not in st.session_state['opt_untagged_df'].columns:
    rng = np.random.default_rng(42)
    now_ts = pd.Timestamp.now()  # include time component
    # Creation between 1 month and 18 months ago, add random time-of-day
    n = len(st.session_state['opt_untagged_df'])
    days_back = rng.integers(low=30, high=540, size=n)
    seconds_in_day = 24 * 3600
    secs_back = rng.integers(low=0, high=seconds_in_day, size=n)
    st.session_state['opt_untagged_df'] = st.session_state['opt_untagged_df'].copy()
    st.session_state['opt_untagged_df']['Date_of_creation'] = [
        (now_ts - pd.Timedelta(int(d), unit='D') - pd.Timedelta(int(s), unit='s')) for d, s in zip(days_back, secs_back)
    ]

# Apply cloud filter first
if cloud_choice_opt == "AWS":
    base_untagged_opt = st.session_state['opt_untagged_df'][st.session_state['opt_untagged_df']['Cloud'] == 'AWS']
elif cloud_choice_opt == "Databricks":
    base_untagged_opt = st.session_state['opt_untagged_df'][st.session_state['opt_untagged_df']['Cloud'] == 'Databricks']
else:
    base_untagged_opt = st.session_state['opt_untagged_df']

# Compute overlap with selected date range and idle_time from creation to range end, bounded by range
start_dt_opt, end_dt_opt = st.session_state['opt_date_range']
dfu = base_untagged_opt.copy()

def _compute_idle_display(row):
    created = pd.to_datetime(row['Date_of_creation'])
    # Idle time from creation up to end of selected range to vary by resource creation time
    end_effective = pd.to_datetime(end_dt_opt)
    seconds = max(0, int((end_effective - created).total_seconds())) if created <= end_effective else 0
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    # Ensure some hours appear when there is any overlap
    if seconds > 0 and hours == 0:
        hours = 1
    return seconds, f"{days}d {hours}h"

secs, disp = zip(*dfu.apply(_compute_idle_display, axis=1)) if not dfu.empty else ([], [])
if not dfu.empty:
    dfu['__overlap_seconds'] = list(secs)
    dfu['idle_time'] = list(disp)
else:
    dfu['__overlap_seconds'] = []
    dfu['idle_time'] = []

# Keep resources that overlap the selected range: created date on/before range end
dfu = dfu[dfu['Date_of_creation'] <= end_dt_opt].copy()
dfu = dfu[dfu['__overlap_seconds'] > 0].copy()

# Display requested columns
display_untagged_opt = dfu[[
    'Resource Name', 'Type', 'Cloud', 'Date_of_creation', 'idle_time', '__overlap_seconds'
]].sort_values('__overlap_seconds', ascending=False)

# Reset index to start from 0 when viewing All clouds
if cloud_choice_opt == "All":
    display_untagged_opt = display_untagged_opt.reset_index(drop=True)

# Hide the helper column from display
display_untagged_opt = display_untagged_opt.drop(columns=['__overlap_seconds'])

st.table(display_untagged_opt)

# Summary metrics reflect filter
col_u1, col_u2, col_u3 = st.columns(3)
with col_u1:
    st.metric("Total Untagged", len(display_untagged_opt))
with col_u2:
    st.metric("AWS Resources", len(display_untagged_opt[display_untagged_opt['Cloud'] == 'AWS']))
with col_u3:
    st.metric("Databricks Resources", len(display_untagged_opt[display_untagged_opt['Cloud'] == 'Databricks']))

# Untagged Spend KPI Cards
u_col1, u_col2 = st.columns(2)

# Parse untagged spend from filtered dfu
currency_symbol = '₹' if (('Cost' in dfu.columns) and dfu['Cost'].astype(str).str.contains('₹').any()) else '$'
untagged_spend_val = 0.0
if 'Cost' in dfu.columns and not dfu.empty:
    untagged_spend_val = float(pd.to_numeric(dfu['Cost'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce').fillna(0).sum())

# Get total spend baseline from utils (category totals per cloud)
aws_costs, db_costs = generate_cloud_spend_data()
total_spend_val = 0.0
if cloud_choice_opt == 'AWS':
    total_spend_val = float(sum(aws_costs.values()))
elif cloud_choice_opt == 'Databricks':
    total_spend_val = float(sum(db_costs.values()))
else:
    total_spend_val = float(sum(aws_costs.values()) + sum(db_costs.values()))

untagged_pct = (untagged_spend_val / total_spend_val * 100.0) if total_spend_val > 0 else np.nan

# Badge color thresholds
def pct_badge(p):
    if np.isnan(p):
        return 'badge-orange', 'N/A'
    if p <= 5:
        return 'badge-green', 'Low'
    if p <= 15:
        return 'badge-orange', 'Moderate'
    return 'badge-red', 'High'

badge_cls_pct, badge_text_pct = pct_badge(untagged_pct)

month_label = pd.to_datetime(end_dt_opt).strftime('%b %Y')
cloud_label = cloud_choice_opt

with u_col1:
    value_fmt = f"{currency_symbol}{untagged_spend_val:,.0f}"
    tooltip_val = f"Untagged Spend: {value_fmt}\nMonth: {month_label}\nCloud: {cloud_label}"
    st.markdown(f"""
    <div class='kpi-card' title="{tooltip_val}" style="background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);">
      <div class='kpi-head' style='display:flex;justify-content:space-between;align-items:center;'>
        <div class='kpi-title' style='font-weight:700;'>Untagged Spend – Value</div>
      </div>
      <div class='kpi-value' style='font-size:2rem;font-weight:800;margin-top:8px;'>{value_fmt}</div>
      <div class='kpi-context' style='opacity:0.9;'>Based on Untagged Resources table</div>
    </div>
    """, unsafe_allow_html=True)

with u_col2:
    pct_str = (f"{untagged_pct:.1f}%" if not np.isnan(untagged_pct) else "N/A")
    tooltip_pct = (
        f"Untagged Spend %: {pct_str}\n"
        f"Untagged Spend: {currency_symbol}{untagged_spend_val:,.0f}\n"
        f"Total Spend: {currency_symbol}{total_spend_val:,.0f}\n"
        f"Month: {month_label} · Cloud: {cloud_label}"
    )
    st.markdown(f"""
    <div class='kpi-card' title="{tooltip_pct}">
      <div class='kpi-head' style='display:flex;justify-content:space-between;align-items:center;'>
        <div class='kpi-title' style='font-weight:700;'>Untagged Spend – %</div>
        <div class='kpi-badge {badge_cls_pct}'>{badge_text_pct}</div>
      </div>
      <div class='kpi-value' style='font-size:2rem;font-weight:800;margin-top:8px;'>{pct_str}</div>
      <div class='kpi-context' style='opacity:0.9;'>Out of total cloud spend</div>
    </div>
    """, unsafe_allow_html=True)

# Optimization Recommendations (moved from Engineering view)
st.markdown("---")
st.subheader("💡 Optimization Recommendations")

recommendations = [
    {
        "Resource": "EC2 m5.xlarge (i-1234567890abcdef0)",
        "Issue": "Underutilized (CPU < 20% for 7 days)",
        "Recommendation": "Downsize to m5.large",
        "Savings": "$45/month",
        "Effort": "Low",
        "Priority": "High"
    },
    {
        "Resource": "RDS db.r5.large (database-1)",
        "Issue": "Storage auto-scaling disabled",
        "Recommendation": "Enable storage auto-scaling",
        "Savings": "$120/month",
        "Effort": "Medium",
        "Priority": "Medium"
    },
    {
        "Resource": "Lambda function (data-processor)",
        "Issue": "High execution time (15s avg)",
        "Recommendation": "Optimize code or increase memory",
        "Savings": "$85/month",
        "Effort": "High",
        "Priority": "Low"
    },
    {
        "Resource": "EKS Cluster (production-cluster)",
        "Issue": "Node autoscaling too aggressive",
        "Recommendation": "Adjust scaling policies",
        "Savings": "$210/month",
        "Effort": "Medium",
        "Priority": "High"
    },
    {
        "Resource": "S3 Bucket (logs-archive)",
        "Issue": "No lifecycle policy",
        "Recommendation": "Add lifecycle rule to transition to Glacier",
        "Savings": "$320/month",
        "Effort": "Low",
        "Priority": "High"
    }
]

st.dataframe(
    pd.DataFrame(recommendations),
    use_container_width=True,
    column_config={
        "Resource": "Resource",
        "Issue": "Issue",
        "Recommendation": "Recommendation",
        "Savings": st.column_config.TextColumn("Monthly Savings"),
        "Effort": st.column_config.SelectboxColumn(
            "Effort",
            options=["Low", "Medium", "High"],
            width="small"
        ),
        "Priority": st.column_config.SelectboxColumn(
            "Priority",
            options=["Low", "Medium", "High"],
            width="small"
        )
    },
    hide_index=True,
)

# Sample data generation
@st.cache_data
def generate_optimization_data():
    # Savings opportunities
    opportunities = [
        {
            "Category": "Compute",
            "Opportunity": "Reserved Instances",
            "Potential Savings": 12500,
            "Effort": "Low",
            "ROI": "High",
            "Resources": 45,
            "Time to Implement": "1 week"
        },
        {
            "Category": "Storage",
            "Opportunity": "S3 Intelligent Tiering",
            "Potential Savings": 8500,
            "Effort": "Low",
            "ROI": "Very High",
            "Resources": 120,
            "Time to Implement": "3 days"
        },
        {
            "Category": "Database",
            "Opportunity": "Aurora Serverless v2",
            "Potential Savings": 6800,
            "Effort": "Medium",
            "ROI": "High",
            "Resources": 12,
            "Time to Implement": "2 weeks"
        },
        {
            "Category": "Compute",
            "Opportunity": "Spot Instances",
            "Potential Savings": 15000,
            "Effort": "High",
            "ROI": "Very High",
            "Resources": 85,
            "Time to Implement": "3 weeks"
        },
        {
            "Category": "Networking",
            "Opportunity": "Direct Connect",
            "Potential Savings": 5200,
            "Effort": "High",
            "ROI": "Medium",
            "Resources": 8,
            "Time to Implement": "4 weeks"
        }
    ]
    
    # Implementation progress
    progress = [
        {"Week": 1, "Planned": 15, "Actual": 12, "Target": 20},
        {"Week": 2, "Planned": 35, "Actual": 28, "Target": 40},
        {"Week": 3, "Planned": 60, "Actual": 52, "Target": 60},
        {"Week": 4, "Planned": 80, "Actual": 75, "Target": 80},
        {"Week": 5, "Planned": 100, "Actual": 0, "Target": 100}
    ]
    
    # Team performance
    team = [
        {"Member": "Alex", "Role": "Cloud Architect", "Tasks Completed": 12, "Impact": "$8,200"},
        {"Member": "Jamie", "Role": "DevOps Engineer", "Tasks Completed": 18, "Impact": "$12,500"},
        {"Member": "Taylor", "Role": "SRE", "Tasks Completed": 9, "Impact": "$5,700"},
        {"Member": "Morgan", "Role": "FinOps Analyst", "Tasks Completed": 15, "Impact": "$9,800"}
    ]
    
    return {
        'opportunities': opportunities,
        'progress': progress,
        'team': team
    }

# Generate data
data = generate_optimization_data()

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Potential Savings", "$48,000", "12.5%")
with col2:
    st.metric("Quick Wins", "23", "5")
with col3:
    st.metric("ROI Estimate", "340%", "12%")
with col4:
    st.metric("Resources Impacted", "270", "18%")

# Savings Opportunities
st.subheader("💰 Savings Opportunities")

# Group by category
category_totals = {}
for opp in data['opportunities']:
    category = opp['Category']
    if category not in category_totals:
        category_totals[category] = 0
    category_totals[category] += opp['Potential Savings']

# Create treemap
fig_treemap = px.treemap(
    data['opportunities'],
    path=['Category', 'Opportunity'],
    values='Potential Savings',
    color='ROI',
    color_discrete_map={
        'Very High': '#2ecc71',
        'High': '#3498db',
        'Medium': '#f39c12',
        'Low': '#e74c3c'
    },
    title='Savings Opportunities by Category and ROI'
)
st.plotly_chart(fig_treemap, use_container_width=True)

# Implementation Progress
st.subheader("📊 Implementation Progress")

# Progress chart
df_progress = pd.DataFrame(data['progress'])

fig_progress = go.Figure()

fig_progress.add_trace(go.Scatter(
    x=df_progress['Week'],
    y=df_progress['Planned'],
    mode='lines+markers',
    name='Planned',
    line=dict(color='#3498db', dash='dash')
))

fig_progress.add_trace(go.Scatter(
    x=df_progress['Week'],
    y=df_progress['Actual'],
    mode='lines+markers',
    name='Actual',
    line=dict(color='#2ecc71')
))

fig_progress.add_trace(go.Scatter(
    x=df_progress['Week'],
    y=df_progress['Target'],
    mode='lines',
    name='Target',
    line=dict(color='#e74c3c', dash='dot')
))

fig_progress.update_layout(
    title='Optimization Implementation Progress',
    xaxis_title='Week',
    yaxis_title='Completion (%)',
    showlegend=True,
    height=400
)

st.plotly_chart(fig_progress, use_container_width=True)

# Team Performance
st.subheader("👥 Team Performance")

col5, col6 = st.columns(2)

with col5:
    # Team metrics
    st.markdown("### Team Impact")
    df_team = pd.DataFrame(data['team'])
    
    # Calculate average impact per task
    df_team['Impact Value'] = df_team['Impact'].str.replace('$', '').str.replace(',', '').astype(float)
    df_team['Impact per Task'] = df_team['Impact Value'] / df_team['Tasks Completed']
    
    # Create a horizontal bar chart for tasks completed
    fig_tasks = px.bar(
        df_team,
        x='Tasks Completed',
        y='Member',
        orientation='h',
        title='Tasks Completed by Team Member',
        text='Tasks Completed',
        color='Role',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_tasks.update_traces(textposition='outside')
    st.plotly_chart(fig_tasks, use_container_width=True)

with col6:
    # Impact per task
    st.markdown("### Impact per Task")
    fig_impact = px.bar(
        df_team,
        x='Impact per Task',
        y='Member',
        orientation='h',
        title='Average Impact per Task',
        text=df_team['Impact per Task'].apply(lambda x: f"${x:,.0f}"),
        color='Role',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_impact.update_traces(textposition='outside')
    st.plotly_chart(fig_impact, use_container_width=True)

# Detailed Opportunities Table
st.subheader("🔍 Detailed Opportunities")

opportunities_df = pd.DataFrame(data['opportunities'])
st.dataframe(
    opportunities_df,
    use_container_width=True,
    column_config={
        "Category": "Category",
        "Opportunity": "Opportunity",
        "Potential Savings": st.column_config.NumberColumn(
            "Potential Savings ($)",
            format="$%,.0f"
        ),
        "Effort": st.column_config.SelectboxColumn(
            "Effort",
            options=["Low", "Medium", "High"],
            width="small"
        ),
        "ROI": st.column_config.SelectboxColumn(
            "ROI",
            options=["Low", "Medium", "High", "Very High"],
            width="small"
        ),
        "Resources": "# Resources",
        "Time to Implement": "Time to Implement"
    },
    hide_index=True,
)

# Action Planning
st.subheader("📅 Action Plan")

# Create a Gantt chart for action items
action_items = [
    {"Task": "Reserved Instances Purchase", "Start": "2025-08-15", "End": "2025-08-22", "Status": "Planned"},
    {"Task": "S3 Lifecycle Policies", "Start": "2025-08-18", "End": "2025-08-25", "Status": "In Progress"},
    {"Task": "Aurora Migration", "Start": "2025-08-25", "End": "2025-09-08", "Status": "Not Started"},
    {"Task": "Spot Fleet Implementation", "Start": "2025-09-01", "End": "2025-09-15", "Status": "Not Started"},
    {"Task": "Direct Connect Setup", "Start": "2025-09-10", "End": "2025-10-01", "Status": "Not Started"}
]

# Convert to DataFrame
df_gantt = pd.DataFrame(action_items)

# Create Gantt chart
fig_gantt = px.timeline(
    df_gantt,
    x_start="Start",
    x_end="End",
    y="Task",
    color="Status",
    title="Optimization Implementation Timeline",
    color_discrete_map={
        "Completed": "#2ecc71",
        "In Progress": "#3498db",
        "Planned": "#f39c12",
        "Not Started": "#e0e0e0"
    }
)

# Update layout
fig_gantt.update_yaxes(autorange="reversed")
fig_gantt.update_layout(
    height=300,
    showlegend=True,
    xaxis_title="",
    yaxis_title=""
)

st.plotly_chart(fig_gantt, use_container_width=True)
