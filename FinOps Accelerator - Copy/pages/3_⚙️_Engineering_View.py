import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import load_css, generate_idle_resources

# Set page config
st.set_page_config(
    page_title="Engineering View | FinOps Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Load CSS
load_css()

# Page title
st.markdown('<h1 class="main-header">⚙️ Engineering View</h1>', unsafe_allow_html=True)

# Minimal CSS for KPI cards (matching Basic Assessment)
st.markdown("""
<style>
  .kpi-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; box-sizing: border-box; }
  .kpi-card h3 { margin: 0; }
</style>
""", unsafe_allow_html=True)

# Cloud selection (for filtering Idle KPIs section)
cloud_choice_eng = st.radio(
    "Cloud selection",
    options=["All", "AWS", "Databricks"],
    horizontal=True,
    key="eng_cloud_filter",
    help="Filter Idle KPIs and table by cloud."
)

# Date range selection (filters Idle KPIs/table when a date column exists)
col_sd_eng, col_ed_eng = st.columns(2)
with col_sd_eng:
    start_date_eng = st.date_input("Start date", value=pd.Timestamp.today().normalize().replace(day=1), key="eng_start_date")
with col_ed_eng:
    end_of_month_eng = (pd.Timestamp.today().normalize().to_period('M').to_timestamp('M'))
    end_date_eng = st.date_input("End date", value=end_of_month_eng, key="eng_end_date")

if pd.to_datetime(end_date_eng) < pd.to_datetime(start_date_eng):
    start_date_eng, end_date_eng = end_date_eng, start_date_eng

st.session_state['eng_date_range'] = (pd.to_datetime(start_date_eng), pd.to_datetime(end_date_eng))

# Helpers: detect a date column and filter by date range if present (Engineering view)
def _detect_date_col_eng(df: pd.DataFrame):
    for col in ['date', 'Date', 'timestamp', 'Timestamp', 'billing_date', 'BillingDate', 'month', 'Month']:
        if col in df.columns:
            return col
    return None

def _filter_df_by_date_eng(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    col = _detect_date_col_eng(df)
    if not col:
        return df
    ser = pd.to_datetime(df[col], errors='coerce')
    mask = (ser >= pd.to_datetime(start_dt)) & (ser <= pd.to_datetime(end_dt))
    return df[mask].copy()

# Spend KPI (same size/style as Finance View)
# Inject Finance-style KPI CSS (scoped class names won't clash)
st.markdown("""
<style>
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

def _eng_status_badge(ratio: float):
    if ratio < 0.8:
        return 'badge-green', 'Within Budget'
    if ratio <= 1.0:
        return 'badge-orange', 'Near Budget'
    return 'badge-red', 'Over Budget'

def _render_eng_spend_kpi(title: str, value_str: str, context: str, ratio: float):
    badge_cls, badge_text = _eng_status_badge(ratio)
    st.markdown(f"""
    <div class='kpi-card-fin'>
      <div class='kpi-head'>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-badge {badge_cls}'>{badge_text}</div>
      </div>
      <div class='kpi-value'>{value_str}</div>
      <div class='kpi-context'>{context}</div>
    </div>
    """, unsafe_allow_html=True)

# Build sample monthly data consistent with Finance view
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
aws_spend = [120000, 118000, 125000, 130000, 128000, 135000, 140000, 138000, 142000, 145000, 148000, 150000]
db_spend = [90000, 92000, 95000, 98000, 100000, 105000, 108000, 110000, 112000, 115000, 118000, 120000]
budget = [150000] * 12

current_year = pd.Timestamp.today().year
month_to_num_eng = {m:i for i, m in enumerate(months, start=1)}
df_months_eng = pd.DataFrame({
    'MonthName': months,
    'MonthNum': [month_to_num_eng[m] for m in months],
    'AWS': aws_spend,
    'Databricks': db_spend,
    'Budget': budget
})
df_months_eng['Date'] = pd.to_datetime({'year': [current_year]*12, 'month': df_months_eng['MonthNum'], 'day': 1})

# Apply Engineering date filter
eng_start_dt, eng_end_dt = st.session_state['eng_date_range']
mask_eng = (df_months_eng['Date'] >= pd.to_datetime(eng_start_dt)) & (df_months_eng['Date'] <= pd.to_datetime(eng_end_dt))
df_f_eng = df_months_eng[mask_eng].copy()
if df_f_eng.empty:
    df_f_eng = df_months_eng.copy()

# Cloud filter from Engineering view
eng_cloud = st.session_state.get('eng_cloud_filter', 'All')
if eng_cloud == 'AWS':
    spend_series_eng = df_f_eng['AWS']
    cloud_label_eng = 'AWS'
elif eng_cloud == 'Databricks':
    spend_series_eng = df_f_eng['Databricks']
    cloud_label_eng = 'Databricks'
else:
    spend_series_eng = df_f_eng['AWS'] + df_f_eng['Databricks']
    cloud_label_eng = 'All Clouds'

# Compute spend and show KPI
budget_total_eng = float(df_f_eng['Budget'].sum()) if 'Budget' in df_f_eng.columns else 0.0
actual_spend_eng = float(spend_series_eng.sum())
ratio_eng = (actual_spend_eng / budget_total_eng) if budget_total_eng > 0 else 0.0

# Dynamic KPI title based on cloud selection
if cloud_label_eng == 'AWS':
    kpi_title = "Total AWS Spend"
elif cloud_label_eng == 'Databricks':
    kpi_title = "Total Databricks Spend"
else:
    kpi_title = "Total Cloud Spend"

_render_eng_spend_kpi(
    title=kpi_title,
    value_str=f"${actual_spend_eng:,.0f}",
    context=f"Total Cloud Spend for the selected date range — {cloud_label_eng}",
    ratio=ratio_eng
)

# Idle Resources KPI Cards (ported from Basic Assessment)
st.markdown("---")
st.subheader("⏰ Idle Resource KPIs")

# Generate or retrieve idle resources data
aws_idle_eng, databricks_idle_eng = generate_idle_resources()

# Cache base dataframes for idle resources once
if 'eng_aws_idle_base' not in st.session_state:
    st.session_state['eng_aws_idle_base'] = pd.DataFrame(aws_idle_eng['resources'])
if 'eng_databricks_idle_base' not in st.session_state:
    st.session_state['eng_databricks_idle_base'] = pd.DataFrame(databricks_idle_eng['resources'])

def _prep_idle_display_eng(df_base: pd.DataFrame):
    df_f = _filter_df_by_date_eng(df_base, *st.session_state['eng_date_range'])
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

if cloud_choice_eng == "AWS":
    st.markdown("### AWS Idle Resources")
    disp_df, count_val, idle_days_val, cost_calc = _prep_idle_display_eng(st.session_state['eng_aws_idle_base'])
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{count_val}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${((cost_calc if cost_calc is not None else aws_idle_eng['total_cost'])):,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{idle_days_val} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)
    st.table(disp_df)
elif cloud_choice_eng == "Databricks":
    st.markdown("### Databricks Idle Resources")
    disp_df, count_val, idle_days_val, cost_calc = _prep_idle_display_eng(st.session_state['eng_databricks_idle_base'])
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{count_val}</h3>\n            <p>Idle Resources</p>\n        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>${((cost_calc if cost_calc is not None else databricks_idle_eng['total_cost'])):,}</h3>\n            <p>Total Cost Impact</p>\n        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class=\"kpi-card\">\n            <h3>{idle_days_val} days</h3>\n            <p>Total Idle Time</p>\n        </div>
        """, unsafe_allow_html=True)
    st.table(disp_df)
else:
    disp_df_aws, count_aws, idle_days_aws, cost_aws = _prep_idle_display_eng(st.session_state['eng_aws_idle_base'])
    disp_df_db, count_db, idle_days_db, cost_db = _prep_idle_display_eng(st.session_state['eng_databricks_idle_base'])
    combined_df = pd.concat([disp_df_aws, disp_df_db], ignore_index=True)
    combined_count = count_aws + count_db
    combined_cost = (cost_aws if cost_aws is not None else aws_idle_eng['total_cost']) + (cost_db if cost_db is not None else databricks_idle_eng['total_cost'])
    combined_idle_days = idle_days_aws + idle_days_db
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
    st.table(combined_df)

# Sample data generation
@st.cache_data
def generate_engineering_data():
    # Resource utilization
    hours = [f"{h:02d}:00" for h in range(24)]
    cpu_usage = [np.random.randint(30, 90) for _ in range(24)]
    memory_usage = [np.random.randint(40, 85) for _ in range(24)]
    
    # Resource types
    resource_types = ['EC2', 'RDS', 'Lambda', 'EKS', 'EMR', 'Redshift', 'OpenSearch']
    resource_counts = [120, 45, 85, 12, 8, 5, 3]
    
    # Performance metrics
    performance_metrics = {
        'metric': ['API Latency (ms)', 'Error Rate (%)', 'Throughput (req/s)', 'Cache Hit Ratio (%)'],
        'current': [125, 0.8, 2450, 92],
        'target': [200, 1.0, 2000, 90],
        'unit': ['ms', '%', 'req/s', '%']
    }
    
    return {
        'hours': hours,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'resource_types': resource_types,
        'resource_counts': resource_counts,
        'performance_metrics': performance_metrics
    }

# Generate data
eng_data = generate_engineering_data()

# Spend by Environment
st.subheader("🌐 Spend by Environment")

# Normalization helper for environment tags
def _normalize_env(tag: str) -> str:
    if not isinstance(tag, str):
        return 'other'
    t = tag.strip().lower()
    if t in ['prod', 'production']:
        return 'prod'
    if t in ['dev', 'development']:
        return 'dev'
    if t in ['test', 'testing', 'qa']:
        return 'test'
    if t in ['sandbox', 'sbx']:
        return 'sandbox'
    return 'other'

# Environment allocation per cloud (placeholder for tag-based aggregation)
env_categories = ['dev', 'test', 'prod', 'sandbox']
env_colors = {
    'dev': '#1f77b4',
    'test': '#ff7f0e',
    'prod': '#2ca02c',
    'sandbox': '#9467bd'
}
aws_env_alloc = {'prod': 0.5, 'dev': 0.2, 'test': 0.2, 'sandbox': 0.1}
db_env_alloc  = {'prod': 0.45, 'dev': 0.25, 'test': 0.2, 'sandbox': 0.1}

# Build daily baseline from monthly totals (reuse df_months_eng)
date_idx_env = pd.date_range(start=eng_start_dt, end=eng_end_dt, freq='D')
daily_env = pd.DataFrame({'Date': date_idx_env})
daily_env['MPer'] = daily_env['Date'].dt.to_period('M')

# Map monthly totals to days
monthly_map = df_months_eng.set_index(df_months_eng['Date'].dt.to_period('M'))
daily_env = daily_env.merge(
    monthly_map[[
        'AWS', 'Databricks', 'Budget'
    ]].rename(columns={'AWS': 'AWSMonth', 'Databricks': 'DBMonth'}),
    left_on='MPer', right_index=True, how='left'
)
daily_env['DaysInMonth'] = daily_env['MPer'].dt.days_in_month
daily_env[['AWSMonth','DBMonth','Budget']] = daily_env[['AWSMonth','DBMonth','Budget']].fillna(method='ffill').fillna(method='bfill')
daily_env['AWS_Daily'] = daily_env['AWSMonth'] / daily_env['DaysInMonth']
daily_env['DB_Daily'] = daily_env['DBMonth'] / daily_env['DaysInMonth']

# Split daily spend by environment
for env in env_categories:
    daily_env[f'AWS_{env}'] = daily_env['AWS_Daily'] * aws_env_alloc[env]
    daily_env[f'DB_{env}']  = daily_env['DB_Daily'] * db_env_alloc[env]

# Aggregate for selected date range and cloud toggle
env_totals = {}
for env in env_categories:
    if cloud_choice_eng == 'AWS':
        env_totals[env] = float(daily_env[f'AWS_{env}'].sum())
    elif cloud_choice_eng == 'Databricks':
        env_totals[env] = float(daily_env[f'DB_{env}'].sum())
    else:
        env_totals[env] = float(daily_env[f'AWS_{env}'].sum() + daily_env[f'DB_{env}'].sum())

env_df = pd.DataFrame({
    'Environment': env_categories,
    'Spend': [env_totals[e] for e in env_categories]
})

fig_env = px.pie(
    env_df,
    names='Environment',
    values='Spend',
    title=f"Spend by Environment — {cloud_choice_eng}",
    color='Environment',
    color_discrete_map=env_colors
)
fig_env.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='%{label}: $%{value:,.0f} (%{percent})')
fig_env.update_layout(showlegend=True)
st.plotly_chart(fig_env, use_container_width=True)

# Resource Utilization
st.subheader("🧰 Job Run Efficiency")

# Controls
granularity = st.selectbox(
    "Time Granularity",
    options=["Daily", "Weekly", "Monthly"],
    index=0,
    key="eng_job_granularity"
)

# Build buckets from Engineering date range
job_start, job_end = st.session_state['eng_date_range']
today = pd.Timestamp.today().normalize()

date_idx = pd.date_range(start=job_start, end=job_end, freq='D')
df = pd.DataFrame({
    'Date': date_idx
})

if granularity == 'Daily':
    df['Bucket'] = df['Date']
    bucket_end = df['Bucket']
elif granularity == 'Weekly':
    df['Bucket'] = df['Date'].dt.to_period('W-MON').dt.start_time
    bucket_end = df['Bucket'] + pd.to_timedelta(6, unit='D')
else:  # Monthly
    df['Bucket'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    bucket_end = df['Bucket'].dt.to_period('M').dt.end_time

grp = df[['Bucket']].drop_duplicates().sort_values('Bucket').reset_index(drop=True)
grp['IsFuture'] = bucket_end.groupby(df['Bucket']).max().reset_index(drop=True) > today

# Synthetic job run counts (placeholder for Databricks job logs)
rng = np.random.default_rng(42)
# Use existing Engineering cloud toggle instead of a separate provider control
cloud_sel = st.session_state.get('eng_cloud_filter', 'All')
aws_params = (120, 0.08)        # (avg total jobs per bucket, failure rate)
dbx_params = (150, 0.12)
if cloud_sel == 'AWS':
    avg_total, fail_rate = aws_params
elif cloud_sel == 'Databricks':
    avg_total, fail_rate = dbx_params
else:
    # All = combine AWS + Databricks
    tot = aws_params[0] + dbx_params[0]
    # weighted failure rate by totals
    fr = (aws_params[0]*aws_params[1] + dbx_params[0]*dbx_params[1]) / tot
    avg_total, fail_rate = tot, fr

# Generate actuals for past buckets
past_count = (~grp['IsFuture']).sum()
totals_past = np.clip(rng.normal(avg_total, avg_total*0.15, size=max(past_count, 0)).round().astype(int), 20, None)
fails_past = np.clip((totals_past * rng.normal(fail_rate, 0.02, size=max(past_count, 0))).round().astype(int), 0, None)
succ_past = totals_past - fails_past

# Projection for future buckets = mean of past
future_count = grp['IsFuture'].sum()
mean_total = int(np.mean(totals_past)) if past_count > 0 else avg_total
mean_fail = int(np.mean(fails_past)) if past_count > 0 else int(avg_total * fail_rate)
mean_succ = max(mean_total - mean_fail, 0)
succ_future = np.array([mean_succ] * future_count, dtype=int)
fails_future = np.array([mean_fail] * future_count, dtype=int)

# Stitch arrays in bucket order
succ = np.concatenate([succ_past, succ_future]) if future_count else succ_past
fail = np.concatenate([fails_past, fails_future]) if future_count else fails_past
total = succ + fail
eff = np.divide(succ, total, out=np.zeros_like(succ, dtype=float), where=total>0)

plot_df = pd.DataFrame({
    'Bucket': grp['Bucket'],
    'Success': succ,
    'Failure': fail,
    'Total': total,
    'Efficiency': eff,
    'Projection': grp['IsFuture']
})

# Build stacked bar chart
fig_jobs = go.Figure()

# For marker pattern per bar, create arrays aligned with x
pattern_future = plot_df['Projection'].map(lambda p: '.' if p else '').tolist()
opacity_future = plot_df['Projection'].map(lambda p: 0.85 if p else 1.0).tolist()

fig_jobs.add_trace(go.Bar(
    x=plot_df['Bucket'],
    y=plot_df['Success'],
    name='Success',
    marker=dict(color='#10b981', pattern=dict(shape=pattern_future)),
    opacity=None,
    hovertemplate='Success: %{y}<br>Total: %{customdata[0]}<br>Efficiency: %{customdata[1]:.1%}<extra></extra>',
    customdata=np.stack([plot_df['Total'], plot_df['Efficiency']], axis=-1)
))
fig_jobs.add_trace(go.Bar(
    x=plot_df['Bucket'],
    y=plot_df['Failure'],
    name='Failure',
    marker=dict(color='#ef4444', pattern=dict(shape=pattern_future)),
    opacity=None,
    hovertemplate='Failure: %{y}<br>Total: %{customdata[0]}<br>Efficiency: %{customdata[1]:.1%}<extra></extra>',
    customdata=np.stack([plot_df['Total'], plot_df['Efficiency']], axis=-1)
))

fig_jobs.update_layout(
    barmode='stack',
    title=f"Job Run Efficiency — {cloud_sel} ({granularity})",
    xaxis_title=granularity,
    yaxis_title='Job Count',
    legend_title='Status',
    height=480
)

st.plotly_chart(fig_jobs, use_container_width=True)

# # Cost vs Performance
# st.subheader("💰 Cost vs Performance")

# # Sample data for cost vs performance
# services = ['EC2 m5.xlarge', 'RDS db.r5.large', 'Lambda', 'EKS Node', 'EMR', 'Redshift', 'OpenSearch']
# cost_per_hour = [0.192, 0.285, 0.00001667, 0.226, 0.270, 0.25, 0.18]
# performance_score = [85, 90, 95, 88, 82, 92, 89]  # Higher is better

# fig_cost_perf = px.scatter(
#     x=cost_per_hour,
#     y=performance_score,
#     text=services,
#     size=[100] * len(services),
#     labels={'x': 'Cost per Hour ($)', 'y': 'Performance Score'},
#     title='Cost vs Performance by Service'
# )

# # Add reference lines
# fig_cost_perf.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Performance Threshold", 
#                        annotation_position="bottom right")
# fig_cost_perf.add_vline(x=0.2, line_dash="dash", line_color="red", annotation_text="Cost Threshold", 
#                        annotation_position="top right")

# # Highlight optimal services
# fig_cost_perf.add_shape(
#     type="rect",
#     x0=0, y0=85,
#     x1=0.2, y1=100,
#     line=dict(color="LightGreen", width=2, dash="dot"),
#     fillcolor="LightGreen",
#     opacity=0.2,
#     layer="below"
# )

# fig_cost_perf.update_traces(
#     textposition='top center',
#     marker=dict(
#         size=20,
#         color='#1f77b4',
#         opacity=0.8,
#         line=dict(width=1, color='DarkSlateGrey')
#     )
# )

# st.plotly_chart(fig_cost_perf, use_container_width=True)

# Resource Recommendations section moved to Early Optimisation view
