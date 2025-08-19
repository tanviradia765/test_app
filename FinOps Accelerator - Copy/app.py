import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="FinOps Accelerator Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
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
    }
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

# Sample data generation functions
@st.cache_data
def generate_health_metrics():
    """Generate sample health metrics data"""
    return {
        'tagging_compliance': 97,
        'idle_usage_efficiency': 94,
        'connectivity': 75,
        'cost_optimization': 82,
        'weights': {
            'tagging': 40,
            'idle': 30,
            'connectivity': 20,
            'optimization': 10
        }
    }

@st.cache_data
def generate_cloud_spend_data():
    """Generate sample cloud spend data"""
    aws_costs = {
        'Compute': 45000,
        'Storage': 12000,
        'Database': 18000,
        'Networking': 8000,
        'Analytics & Big Data': 15000,
        'AI/ML': 22000,
        'Security & Management': 6000
    }
    
    databricks_costs = {
        'Compute': 35000,
        'Storage': 8000,
        'Workflows & Jobs': 12000,
        'SQL & BI': 18000,
        'Machine Learning': 25000,
        'Networking & Integration': 5000,
        'Governance & Security': 7000
    }
    
    return aws_costs, databricks_costs

@st.cache_data
def generate_idle_resources():
    """Generate sample idle resources data"""
    aws_idle = {
        'count': 12,
        'total_cost': 3200,
        'resources': [
            {'name': 'ec2-web-server-01', 'type': 'EC2', 'cost': 450, 'idle_time': '7 days'},
            {'name': 'rds-analytics-db', 'type': 'RDS', 'cost': 680, 'idle_time': '12 days'},
            {'name': 'lambda-data-processor', 'type': 'Lambda', 'cost': 120, 'idle_time': '5 days'},
            {'name': 'ebs-volume-backup', 'type': 'EBS', 'cost': 200, 'idle_time': '15 days'},
            {'name': 'elb-load-balancer', 'type': 'ELB', 'cost': 300, 'idle_time': '8 days'}
        ]
    }
    
    databricks_idle = {
        'count': 8,
        'total_cost': 2100,
        'resources': [
            {'name': 'cluster-analytics-01', 'type': 'Cluster', 'cost': 800, 'idle_time': '6 days'},
            {'name': 'job-etl-pipeline', 'type': 'Job', 'cost': 450, 'idle_time': '10 days'},
            {'name': 'ml-experiment-cluster', 'type': 'ML Cluster', 'cost': 600, 'idle_time': '4 days'},
            {'name': 'sql-warehouse-dev', 'type': 'SQL Warehouse', 'cost': 250, 'idle_time': '9 days'}
        ]
    }
    
    return aws_idle, databricks_idle

@st.cache_data
def generate_untagged_resources():
    """Generate sample untagged resources data"""
    return pd.DataFrame([
        {'Resource Name': 'ec2-prod-web-01', 'Type': 'EC2 Instance', 'Cloud': 'AWS', 'Cost': '$450/month', 'Missing Tags': 'Environment, Owner'},
        {'Resource Name': 'rds-customer-db', 'Type': 'RDS Database', 'Cloud': 'AWS', 'Cost': '$680/month', 'Missing Tags': 'Project, CostCenter'},
        {'Resource Name': 'cluster-data-science', 'Type': 'Databricks Cluster', 'Cloud': 'Databricks', 'Cost': '$1200/month', 'Missing Tags': 'Team, Environment'},
        {'Resource Name': 's3-backup-bucket', 'Type': 'S3 Bucket', 'Cloud': 'AWS', 'Cost': '$120/month', 'Missing Tags': 'Owner, Retention'},
        {'Resource Name': 'job-daily-etl', 'Type': 'Databricks Job', 'Cloud': 'Databricks', 'Cost': '$300/month', 'Missing Tags': 'Project, Owner'},
        {'Resource Name': 'lambda-api-gateway', 'Type': 'Lambda Function', 'Cloud': 'AWS', 'Cost': '$80/month', 'Missing Tags': 'Environment, Team'},
        {'Resource Name': 'ml-model-endpoint', 'Type': 'ML Endpoint', 'Cloud': 'Databricks', 'Cost': '$500/month', 'Missing Tags': 'CostCenter, Owner'}
    ])

def calculate_health_score(metrics):
    """Calculate the infrastructure health score"""
    weights = metrics['weights']
    tagging_score = metrics['tagging_compliance'] * weights['tagging'] / 100
    idle_score = metrics['idle_usage_efficiency'] * weights['idle'] / 100
    connectivity_score = metrics['connectivity'] * weights['connectivity'] / 100
    optimization_score = metrics['cost_optimization'] * weights['optimization'] / 100
    
    total_score = tagging_score + idle_score + connectivity_score + optimization_score
    
    return total_score, {
        'tagging': tagging_score,
        'idle': idle_score,
        'connectivity': connectivity_score,
        'optimization': optimization_score
    }

def create_health_score_gauge(score):
    """Create a gauge chart for health score"""
    if score >= 90:
        color = "#28a745"
        status = "Excellent"
    elif score >= 70:
        color = "#ffc107"
        status = "Good"
    else:
        color = "#dc3545"
        status = "Needs Attention"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Infrastructure Health Score<br><span style='font-size:0.8em;color:{color}'>{status}</span>"},
        delta = {'reference': 85},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 70], 'color': "lightgray"},
                {'range': [70, 90], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_cost_breakdown_chart(aws_costs, databricks_costs):
    """Create cost breakdown charts for AWS and Databricks"""
    # AWS Pie Chart
    aws_fig = px.pie(
        values=list(aws_costs.values()),
        names=list(aws_costs.keys()),
        title="AWS Cost Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    aws_fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Databricks Pie Chart
    db_fig = px.pie(
        values=list(databricks_costs.values()),
        names=list(databricks_costs.keys()),
        title="Databricks Cost Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    db_fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return aws_fig, db_fig

# Sidebar navigation
st.sidebar.title("🏢 FinOps Accelerator")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Select View",
    ["Basic Assessment", "Finance View", "Engineering View", "Early Optimisation View"]
)

# Main content based on selected page
if page == "Basic Assessment":
    st.markdown('<h1 class="main-header">📊 Basic Assessment View</h1>', unsafe_allow_html=True)
    
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
                f"  <div class='metric-head'><div class='metric-title'>⚠ Idle Usage</div><div class='metric-sub'>Weight: {metrics[1]['Weight']}%</div></div>"
                f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[1]['Score']}%'></span></div>"
                f"  <div class='metric-sub'>Score: {metrics[1]['Score']}%</div>"
                f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[1]['Impact']}<br/><em>{metrics[1]['Notes']}</em></div>"
                f"</div>"
                "</div>", unsafe_allow_html=True)

    # Row 2: Connectivity, Cost Optimization
    st.markdown("<div class='metric-row'>" 
                f"<div class='metric-card-inline'>"
                f"  <div class='metric-head'><div class='metric-title'>🔌 Connectivity</div><div class='metric-sub'>Weight: {metrics[2]['Weight']}%</div></div>"
                f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[2]['Score']}%'></span></div>"
                f"  <div class='metric-sub'>Score: {metrics[2]['Score']}%</div>"
                f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[2]['Impact']}<br/><em>{metrics[2]['Notes']}</em></div>"
                f"</div>"
                f"<div class='metric-card-inline'>"
                f"  <div class='metric-head'><div class='metric-title'>💰 Cost Optimization</div><div class='metric-sub'>Weight: {metrics[3]['Weight']}%</div></div>"
                f"  <div class='metric-bar'><span class='metric-fill' style='width: {metrics[3]['Score']}%'></span></div>"
                f"  <div class='metric-sub'>Score: {metrics[3]['Score']}%</div>"
                f"  <div class='metric-notes'><strong>Impact:</strong> {metrics[3]['Impact']}<br/><em>{metrics[3]['Notes']}</em></div>"
                f"</div>"
                "</div>", unsafe_allow_html=True)

    st.divider()

    # Total Cloud Spend (directly below Health Score)
    st.markdown("---")
    st.subheader("💰 Total Cloud Spend")

    # Cost breakdown charts
    aws_fig, db_fig = create_cost_breakdown_chart(aws_costs, databricks_costs)

    spend_col1, spend_col2 = st.columns(2)
    with spend_col1:
        st.plotly_chart(aws_fig, use_container_width=True)
        st.metric("Total AWS Spend", f"${sum(aws_costs.values()):,}")

    with spend_col2:
        st.plotly_chart(db_fig, use_container_width=True)
        st.metric("Total Databricks Spend", f"${sum(databricks_costs.values()):,}")
    
    # Idle Resources KPI Cards
    st.markdown("---")
    st.subheader("⏰ Idle Resource KPIs")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### AWS Idle Resources")
        # Prepare sorted df and total idle days
        if 'aws_idle_df' not in st.session_state:
            df = pd.DataFrame(aws_idle['resources'])
            df['idle_days'] = df['idle_time'].str.extract(r"(\d+)").astype(int)
            total_idle_days_aws = int(df['idle_days'].sum())
            df = df.sort_values('idle_days', ascending=False).drop(columns=['idle_days'])
            st.session_state['aws_idle_df'] = df
            st.session_state['aws_total_idle_days'] = total_idle_days_aws
        
        # KPI Card with total idle days
        st.markdown(f"""
        <div class="kpi-card">
            <h2>{aws_idle['count']}</h2>
            <p>Idle Resources</p>
            <h3>${aws_idle['total_cost']:,}</h3>
            <p>Total Cost Impact</p>
            <h3>{st.session_state['aws_total_idle_days']} days</h3>
            <p>Total Idle Time</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AWS Idle resources details (stable, lighter rendering) sorted by idle_time desc
        st.table(st.session_state['aws_idle_df'])
    
    with col4:
        st.markdown("### Databricks Idle Resources")
        # Prepare sorted df and total idle days
        if 'databricks_idle_df' not in st.session_state:
            df_db = pd.DataFrame(databricks_idle['resources'])
            df_db['idle_days'] = df_db['idle_time'].str.extract(r"(\d+)").astype(int)
            total_idle_days_db = int(df_db['idle_days'].sum())
            df_db = df_db.sort_values('idle_days', ascending=False).drop(columns=['idle_days'])
            st.session_state['databricks_idle_df'] = df_db
            st.session_state['db_total_idle_days'] = total_idle_days_db
        
        # KPI Card with total idle days
        st.markdown(f"""
        <div class="kpi-card">
            <h2>{databricks_idle['count']}</h2>
            <p>Idle Resources</p>
            <h3>${databricks_idle['total_cost']:,}</h3>
            <p>Total Cost Impact</p>
            <h3>{st.session_state['db_total_idle_days']} days</h3>
            <p>Total Idle Time</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Databricks Idle resources details (stable, lighter rendering) sorted by idle_time desc
        st.table(st.session_state['databricks_idle_df'])
    
    st.markdown("---")
    st.subheader("🏷️ Untagged Resources")
    st.markdown("Resources missing one or more mandatory tags as per organization's tagging policy:")
    
    # Stable rendering without heavy Styler to avoid flicker
    if 'untagged_df' not in st.session_state:
        st.session_state['untagged_df'] = untagged_df
    st.table(st.session_state['untagged_df'])
    
    # Summary metrics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total Untagged", len(untagged_df))
    with col6:
        st.metric("AWS Resources", len(untagged_df[untagged_df['Cloud'] == 'AWS']))
    with col7:
        st.metric("Databricks Resources", len(untagged_df[untagged_df['Cloud'] == 'Databricks']))
    with col8:
        total_untagged_cost = sum([int(cost.replace('$', '').replace('/month', '')) for cost in untagged_df['Cost']])
        st.metric("Monthly Cost Impact", f"${total_untagged_cost}")

elif page == "Finance View":
    st.markdown('<h1 class="main-header">💼 Finance View</h1>', unsafe_allow_html=True)
    st.info("🚧 Finance View - Coming Soon! This will include detailed financial analytics, budget tracking, and cost forecasting.")
    
    # Placeholder content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Budget", "$150,000", "5.2%")
    with col2:
        st.metric("Current Spend", "$142,300", "-2.1%")
    with col3:
        st.metric("Forecast", "$148,500", "1.8%")

elif page == "Engineering View":
    st.markdown('<h1 class="main-header">⚙️ Engineering View</h1>', unsafe_allow_html=True)
    st.info("🚧 Engineering View - Coming Soon! This will include resource utilization, performance metrics, and technical recommendations.")
    
    # Placeholder content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Utilization", "67%", "3.2%")
    with col2:
        st.metric("Memory Usage", "78%", "-1.5%")
    with col3:
        st.metric("Storage Efficiency", "85%", "2.1%")

elif page == "Early Optimisation View":
    st.markdown('<h1 class="main-header">🚀 Early Optimisation View</h1>', unsafe_allow_html=True)
    st.info("🚧 Early Optimisation View - Coming Soon! This will include optimization recommendations, potential savings, and action items.")
    
    # Placeholder content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Potential Savings", "$12,400", "8.7%")
    with col2:
        st.metric("Quick Wins", "23", "5")
    with col3:
        st.metric("ROI Estimate", "340%", "12%")

# Footer
st.markdown("---")
if 'footer_last_updated' not in st.session_state:
    st.session_state['footer_last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>FinOps Accelerator Dashboard v1.0 | Built with Streamlit & Python</p>
        <p>Last Updated: {st.session_state['footer_last_updated']}</p>
    </div>
    """,
    unsafe_allow_html=True
)
