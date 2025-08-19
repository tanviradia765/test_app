import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Custom CSS for better styling
def load_css():
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
    </style>
    """, unsafe_allow_html=True)

# Sample data generation functions
@st.cache_data
def generate_health_metrics():
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
    aws_fig = px.pie(
        values=list(aws_costs.values()),
        names=list(aws_costs.keys()),
        title="AWS Cost Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    aws_fig.update_traces(textposition='inside', textinfo='percent+label')
    
    db_fig = px.pie(
        values=list(databricks_costs.values()),
        names=list(databricks_costs.keys()),
        title="Databricks Cost Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    db_fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return aws_fig, db_fig
