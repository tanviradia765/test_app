# FinOps Accelerator Dashboard

A production-level FinOps (Financial Operations) dashboard built with Python and Streamlit for cloud cost management and optimization.

## 🚀 Features

### Screen 1: Basic Assessment View
- **Infrastructure Health Score**: Weighted gauge chart showing overall cloud infrastructure fitness
- **Cloud Spend Breakdown**: Detailed cost analysis for AWS and Databricks services
- **Idle Resource KPIs**: Interactive cards showing idle resources with cost impact
- **Untagged Resources**: Table of resources missing mandatory tags

### Screen 2-4: Coming Soon
- **Finance View**: Budget tracking and cost forecasting
- **Engineering View**: Resource utilization and performance metrics
- **Early Optimisation View**: Optimization recommendations and potential savings

## 📊 Health Score Calculation

The Infrastructure Health Score is calculated using a weighted average of:

| Metric | Weight | Description |
|--------|--------|-------------|
| Tagging Compliance | 40% | Percentage of resources with proper tags |
| Idle Usage Efficiency | 30% | Efficiency based on idle resource ratio |
| Provider Connectivity | 20% | Cloud provider connection health |
| Cost Optimization | 10% | Percentage of implemented savings |

**Color Coding:**
- 🟢 90-100%: Excellent
- 🟡 70-89%: Good  
- 🔴 Below 70%: Needs Attention

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard:**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
FinOps Accelerator/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🎯 Key Metrics Displayed

### AWS Cost Categories
- Compute, Storage, Database, Networking
- Analytics & Big Data, AI/ML, Security & Management

### Databricks Cost Categories  
- Compute, Storage, Workflows & Jobs
- SQL & BI, Machine Learning, Networking & Integration
- Governance & Security

### Idle Resource Tracking
- Resource count and type
- Cost impact per resource
- Idle duration tracking

## 🔧 Customization

The dashboard uses sample data for demonstration. To connect real data sources:

1. Replace the `generate_*` functions with actual data connectors
2. Integrate with cloud provider APIs (AWS, Azure, GCP, Databricks)
3. Connect to your organization's tagging policies
4. Customize health score weights based on your priorities

## 📈 Future Enhancements

- Real-time data integration
- Advanced cost forecasting
- Automated optimization recommendations
- Custom alerting and notifications
- Export capabilities for reports

## 🤝 Contributing

This is a production-ready template that can be extended based on your organization's specific FinOps requirements.

---

**Built with:** Python 3.8+, Streamlit, Plotly, Pandas
**Last Updated:** 2025-01-11
