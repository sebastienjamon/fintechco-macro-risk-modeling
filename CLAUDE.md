# Working with Claude Code on the FinTechCo Demo

This guide provides practical instructions for using Claude Code to explore, analyze, and extend the FinTechCo Macro Risk Modeling demonstration environment.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Common Tasks & Example Prompts](#common-tasks--example-prompts)
3. [Data Analysis Workflows](#data-analysis-workflows)
4. [Model Development & Iteration](#model-development--iteration)
5. [Working with External Data Sources](#working-with-external-data-sources)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

This demo is configured with two MCP (Model Context Protocol) servers:
- **FRED MCP Server**: Access to Federal Reserve Economic Data
- **Snowflake MCP Server**: Connection to synthetic payment and fraud data

### Quick Start

1. **Open the project in Claude Code:**
   ```bash
   cd /path/to/fintechco-macro-risk-modeling
   claude
   ```

2. **Verify MCP servers are connected:**
   ```
   Check if FRED and Snowflake MCP servers are available
   ```
   Claude will confirm connectivity to both data sources.

3. **Explore the codebase:**
   ```
   Give me an overview of this project structure and what each component does
   ```

---

## Common Tasks & Example Prompts

### 1. Exploring the Data

#### View Payment Transaction Data
```
Show me a sample of the payment transactions data. What columns are available?
```

#### Analyze Fraud Trends
```
Analyze the fraud incidents data and show me:
- Monthly fraud trends for 2025
- Breakdown by fraud type
- Average loss amount per incident type
```

#### Understand Macroeconomic Context
```
Fetch the latest unemployment rate and federal funds rate from FRED.
How have they changed over the past 12 months?
```

### 2. Running Models

#### Execute ID Card Validation Model
```
Run the ID card validation model and explain the results.
What's the current accuracy and what are the top features?
```

#### Fraud Detection Model Analysis
```
Run the fraud classification model. Show me:
- ROC-AUC score
- Feature importance
- Confusion matrix analysis
```

#### Revenue Prediction
```
Execute the linear regression model and show me how macroeconomic
indicators correlate with revenue predictions
```

### 3. Custom Analysis

#### Correlation Analysis
```
Analyze the correlation between unemployment rate (from FRED) and
synthetic ID fraud incidents. Create a visualization showing this relationship.
```

#### Seasonal Patterns
```
Investigate if there are any seasonal patterns in payment transaction volumes.
Group by month and show me the trends.
```

#### Risk Segmentation
```
Segment customers by fraud risk score and analyze the characteristics
of high-risk vs. low-risk segments
```

### 4. Extending the Models

#### Feature Engineering
```
Add a new feature to the fraud detection model: "transaction_velocity"
which measures how many transactions a customer made in the past 24 hours.
Retrain the model and compare performance.
```

#### Hyperparameter Tuning
```
The fraud detection model is overfitting (train AUC 0.99, test AUC 0.46).
Can you tune the hyperparameters to reduce overfitting? Try adjusting:
- max_depth
- min_samples_split
- class_weight
```

#### Ensemble Approach
```
Create an ensemble model combining Random Forest and Gradient Boosting
for fraud detection. Compare performance to the existing single model.
```

### 5. Visualization & Reporting

#### Create Custom Visualizations
```
Create a dashboard showing:
1. Monthly fraud loss trend
2. Fraud type distribution
3. Correlation between unemployment and fraud incidents
4. Geographic distribution of high-risk transactions
```

#### Generate Executive Summary
```
Analyze the macro risk assessment data and generate an executive summary
highlighting the top 3 risks and recommended actions
```

---

## Data Analysis Workflows

### Workflow 1: Investigating the Synthetic ID Fraud Surge

**Goal:** Understand why synthetic ID fraud increased 338% in June-November 2025

```
I need to investigate the synthetic ID fraud surge mentioned in the
risk assessment. Can you:

1. Query Snowflake for synthetic ID fraud incidents by month
2. Calculate the month-over-month growth rate
3. Analyze if there are common patterns (merchant categories, amounts, locations)
4. Check if there's correlation with any macroeconomic indicators
5. Create visualizations to present findings
```

**Expected Output:**
- Time series plot of synthetic ID incidents
- Statistical analysis of growth patterns
- Hypothesis about root causes
- Recommendations for further investigation

---

### Workflow 2: Building a Macro Stress Indicator

**Goal:** Create a composite indicator combining unemployment, inflation, and Fed policy

```
Help me build a "Macro Stress Score" that combines:
- Unemployment rate (higher = more stress)
- CPI inflation rate (higher = more stress)
- Federal funds rate change (rapid changes = more stress)

Weight them appropriately and create a time series showing the stress
score from 2024-2025. Then correlate it with fraud incidents.
```

**Expected Output:**
- Python function to calculate composite score
- Time series visualization
- Correlation analysis with fraud data
- Threshold recommendations for risk alerts

---

### Workflow 3: Customer Churn Prediction

**Goal:** Predict which customers are likely to churn based on behavior

```
Using the customer_metrics and sentiment data:

1. Create features: transaction frequency, spending volatility, sentiment scores
2. Define churn as: no transactions in the last 60 days
3. Train a classification model to predict churn
4. Identify top factors driving churn
5. Generate a list of at-risk customers for retention efforts
```

**Expected Output:**
- Feature engineering pipeline
- Trained classification model
- Feature importance analysis
- CSV file with at-risk customers and churn probabilities

---

## Model Development & Iteration

### Iterative Improvement Process

Claude Code excels at iterative model development. Here's a recommended workflow:

#### Step 1: Baseline Model
```
Create a baseline fraud detection model using just 5 features:
- transaction_amount
- merchant_category
- transaction_type
- hour_of_day
- customer_avg_amount

Show me the ROC-AUC score.
```

#### Step 2: Feature Engineering
```
The baseline model achieved 0.72 AUC. Add these engineered features:
- amount_deviation (how far from customer average)
- transaction_count_last_24h (velocity)
- is_high_risk_merchant (flag for risky categories)
- is_unusual_time (late night/early morning)

Retrain and compare performance.
```

#### Step 3: Address Class Imbalance
```
The model has class imbalance (fraud rate: 1.5%). Apply SMOTE to
oversample the minority class. Does this improve recall without
hurting precision too much?
```

#### Step 4: Explainability
```
Add SHAP values to explain individual predictions. For the top 10
highest-risk transactions, show me which features contributed most
to the fraud prediction.
```

#### Step 5: Production Readiness
```
Package this model for production:
1. Create a predict() function that takes raw transaction data
2. Add input validation
3. Include confidence intervals
4. Log predictions to a CSV file
5. Add error handling
```

---

## Working with External Data Sources

### FRED MCP Server

The FRED MCP server provides access to 817,000+ economic time series.

#### Search for Indicators
```
Search FRED for economic indicators related to consumer credit and debt
```

#### Fetch Specific Series
```
Get the consumer confidence index (series: UMCSENT) for the past 2 years
```

#### Correlate with Internal Data
```
Fetch the consumer price index from FRED and correlate it with our
internal transaction volume data. Are customers spending less when
inflation is high?
```

#### Available FRED Functions
- `fred_search`: Search for data series by keyword
- `fred_browse`: Browse categories, releases, or sources
- `fred_get_series`: Retrieve time series data with transformations

### Snowflake MCP Server

The Snowflake MCP server connects to the demo database with synthetic data.

#### Query Snowflake Directly
```
Query Snowflake to find all high-value transactions (>$5000) in
November 2025 that were flagged as fraudulent
```

#### Create Custom Views
```
Create a Snowflake view that aggregates fraud incidents by week
and merchant category
```

#### Available Snowflake Functions
- `run_snowflake_query`: Execute custom SQL queries
- `list_objects`: List databases, schemas, tables
- `describe_object`: Get schema details for a table
- `create_object`: Create new database objects

#### Example Queries

**Fraud Rate by Merchant Category:**
```sql
SELECT
    merchant_category,
    COUNT(*) as total_transactions,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) as fraud_rate_percent
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS t
JOIN DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS f
    ON t.transaction_id = f.transaction_id
GROUP BY merchant_category
ORDER BY fraud_rate_percent DESC
```

**Monthly Revenue Trend:**
```sql
SELECT
    DATE_TRUNC('month', transaction_date) as month,
    SUM(amount) as monthly_revenue,
    COUNT(*) as transaction_count
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS
WHERE status = 'completed'
GROUP BY month
ORDER BY month
```

---

## Best Practices

### 1. Start with Exploration

Before diving into modeling, ask Claude to:
- Summarize the data schema
- Show sample data
- Identify missing values or data quality issues
- Visualize distributions

**Example:**
```
Before we build a model, help me understand the fraud_incidents table:
- How many records?
- What's the date range?
- Any missing values?
- Show the distribution of fraud_type
```

### 2. Be Specific About Requirements

Claude performs best with clear, specific requirements:

❌ **Vague:** "Improve the fraud model"

✅ **Specific:** "The fraud model has 99% train AUC but only 46% test AUC. This indicates overfitting. Can you reduce max_depth from 15 to 8, increase min_samples_split from 10 to 50, and retrain? Show me the before/after metrics."

### 3. Iterate Incrementally

Break complex tasks into smaller steps:

**Instead of:**
```
Build a complete fraud detection system with real-time scoring,
model retraining, alerting, and a dashboard
```

**Try:**
```
Step 1: Build a baseline fraud model using Random Forest
[Wait for result]

Step 2: Add feature engineering for transaction velocity
[Wait for result]

Step 3: Create a simple scoring function
[Continue...]
```

### 4. Request Explanations

Ask Claude to explain its reasoning:
```
You chose XGBoost over Random Forest for this task. Can you explain why
and what are the tradeoffs?
```

### 5. Validate Results

Always ask Claude to validate model outputs:
```
The model achieved 99% accuracy. Can you:
- Check if this is due to class imbalance
- Calculate precision, recall, and F1 score
- Show the confusion matrix
- Validate on a held-out test set
```

### 6. Document as You Go

Request documentation for production readiness:
```
Create a README.md for the fraud detection model explaining:
- What it does
- Input/output format
- Performance metrics
- How to retrain
- Known limitations
```

---

## Best Practices for This Demo

### Dos ✅

1. **Explore relationships between FRED data and internal metrics**
   ```
   How does the unemployment rate correlate with our churn rate?
   ```

2. **Generate synthetic scenarios**
   ```
   Simulate what would happen to our revenue if unemployment rises to 5.5%
   ```

3. **Compare model versions**
   ```
   Compare the performance of the current fraud model vs. a Gradient Boosting
   approach. Which is better for our use case?
   ```

4. **Ask for production considerations**
   ```
   What would we need to consider to deploy this model to production?
   ```

5. **Request visualizations**
   ```
   Create an interactive plotly dashboard showing fraud trends over time
   ```

### Don'ts ❌

1. **Don't assume data relationships without validation**
   ```
   ❌ "Create a model assuming unemployment causes fraud"
   ✅ "Analyze if there's a correlation between unemployment and fraud,
       then build a model if the relationship is statistically significant"
   ```

2. **Don't skip data quality checks**
   ```
   ❌ "Train a model on the transactions table"
   ✅ "First check the transactions table for missing values, outliers,
       and data quality issues, then train a model"
   ```

3. **Don't ignore class imbalance**
   ```
   ❌ "The model has 99% accuracy, looks great!"
   ✅ "Check if the 99% accuracy is due to class imbalance. Calculate
       precision, recall, and ROC-AUC"
   ```

4. **Don't deploy without testing**
   ```
   ❌ "This model looks good, let's deploy it"
   ✅ "Test this model on November 2025 data (held-out set) before
       considering deployment"
   ```

---

## Example Analysis Sessions

### Session 1: Economic Stress Analysis

```
USER: I want to understand if the current economic conditions are
creating stress for our customers.

CLAUDE: I'll help you analyze economic stress indicators. Let me:
1. Fetch recent unemployment rate from FRED
2. Get CPI inflation data
3. Query our transaction data for spending patterns
4. Correlate external and internal metrics

[Claude executes analysis...]

Based on the analysis:
- Unemployment rose from 3.7% to 4.4% (approaching Sahm Rule threshold)
- CPI inflation re-accelerated from 2.3% to 3.0%
- Discretionary spending ratio declined from 0.96 to 0.90
- Travel spending dropped 14.3% QoQ

This suggests stagflation pressure. Would you like me to:
- Build a macro stress indicator?
- Identify at-risk customer segments?
- Create a monitoring dashboard?

USER: Yes, build a macro stress indicator and identify at-risk customers.

[Session continues...]
```

### Session 2: Model Performance Investigation

```
USER: The fraud model shows signs of overfitting. Can you diagnose and fix it?

CLAUDE: I see the issue - training ROC-AUC is 0.9943 but test is only 0.4583.
Let me investigate:

1. Checking class imbalance... Fraud rate is 1.5% (highly imbalanced)
2. Examining model complexity... max_depth=15 may be too deep
3. Reviewing features... 21 features for 50K samples is reasonable

Recommendations:
1. Add class_weight='balanced' to handle imbalance
2. Reduce max_depth from 15 to 8
3. Increase min_samples_split from 10 to 20
4. Consider SMOTE for minority class oversampling

Would you like me to implement these changes?

USER: Yes, implement and show before/after metrics.

[Claude implements changes and shows comparison...]
```

### Session 3: Production Deployment Planning

```
USER: We need to deploy the ID validation model to production. What do we need?

CLAUDE: For production deployment of the ID validation model, you'll need:

**Infrastructure:**
- GPU instances (NVIDIA A10G or better) for real-time inference
- Load balancer for high availability
- Model serving platform (TensorFlow Serving, TorchServe, or ONNX Runtime)

**Monitoring:**
- Latency tracking (target: <500ms p99)
- False positive rate monitoring
- Model drift detection
- Error rate alerting

**Testing:**
- Load testing (1000 req/min sustained)
- Shadow mode deployment initially
- A/B testing framework

**Compliance:**
- Audit logging for all predictions
- GDPR/CCPA data retention policy
- Explainability reports for rejections

Would you like me to create:
1. Infrastructure as Code (Terraform)?
2. Monitoring dashboard configuration?
3. Deployment documentation?

[Continue based on user choice...]
```

---

## Troubleshooting

### Common Issues

#### Issue: Claude can't find a file

**Symptom:**
```
Error: File not found: data/synthetic/payment_transactions.csv
```

**Solution:**
```
Check what files exist in the data/synthetic directory and show me the project structure
```

#### Issue: Snowflake connection error

**Symptom:**
```
Error: Could not connect to Snowflake database
```

**Solution:**
1. Verify MCP server configuration in `.mcp.json`
2. Check Snowflake credentials are valid
3. Ask Claude: `Test the Snowflake connection by listing available databases`

#### Issue: FRED data fetch fails

**Symptom:**
```
Error: Series not found: UNRATES
```

**Solution:**
```
Search FRED for unemployment rate series. The correct series ID is UNRATE (not UNRATES)
```

#### Issue: Model training takes too long

**Symptom:**
Model training doesn't complete within reasonable time

**Solution:**
```
The model training is slow. Can you:
1. Check the dataset size
2. Reduce n_estimators if using ensemble methods
3. Add a progress bar
4. Consider using a smaller sample for development
```

#### Issue: Out of memory error

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```
The dataset is too large for memory. Can you:
1. Check the size of the dataset
2. Implement batch processing
3. Use chunking for file I/O
4. Sample the data for development
```

---

## Advanced Use Cases

### 1. Custom Risk Scoring Model

```
Build a custom risk scoring model that combines:
- Traditional fraud model output (40% weight)
- Macro economic stress indicator (30% weight)
- Customer sentiment score (20% weight)
- Transaction velocity anomaly (10% weight)

Create a final risk score from 0-100 and categorize as:
- Low risk: 0-30
- Medium risk: 31-60
- High risk: 61-80
- Critical risk: 81-100

Test it on November 2025 data and show me the distribution.
```

### 2. Scenario Planning

```
Run scenario analysis for Q1 2026:

Scenario 1 (Base Case):
- Unemployment stays at 4.4%
- Inflation declines to 2.5%
- Fed cuts rates by 25bps

Scenario 2 (Recession):
- Unemployment rises to 5.5%
- Inflation at 2.0%
- Fed cuts rates by 100bps

Scenario 3 (Stagflation):
- Unemployment rises to 5.0%
- Inflation rises to 4.0%
- Fed holds rates steady

For each scenario, predict:
- Expected fraud incident count
- Revenue impact
- Customer churn rate
```

### 3. Automated Report Generation

```
Create an automated weekly risk report that:
1. Fetches latest FRED economic data
2. Queries Snowflake for past week's fraud incidents
3. Calculates key metrics (fraud rate, loss amount, detection rate)
4. Compares to previous week
5. Highlights anomalies
6. Generates a Markdown report with visualizations
7. Exports to PDF

Schedule this to run every Monday at 9am.
```

---

## Tips for Data Scientists

### Exploratory Data Analysis (EDA)

Claude can accelerate EDA significantly:

```
Perform comprehensive EDA on the payment transactions data:
- Summary statistics
- Missing value analysis
- Distribution plots for numeric features
- Frequency tables for categorical features
- Correlation matrix
- Outlier detection
- Temporal patterns (day of week, hour of day)
- Geographic patterns if applicable

Save all visualizations to docs/eda/ folder.
```

### Feature Engineering Pipeline

```
Create a feature engineering pipeline for fraud detection:

1. Temporal features:
   - hour_of_day, day_of_week, is_weekend, is_night

2. Customer behavior features:
   - customer_avg_amount (rolling 30 days)
   - customer_transaction_count (rolling 7 days)
   - amount_deviation (z-score from customer mean)

3. Transaction pattern features:
   - is_high_value (>$1000)
   - is_unusual_merchant (merchant category != customer's typical)
   - velocity_flag (>5 transactions in 1 hour)

4. External data features:
   - unemployment_rate (join with FRED data by month)
   - inflation_rate (join with FRED CPI data)

Package this as a reusable transform_features() function.
```

### Model Comparison Framework

```
Create a model comparison framework that trains and evaluates:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM

For each model, calculate:
- ROC-AUC
- Precision, Recall, F1
- Confusion matrix
- Training time
- Inference time

Generate a comparison table and recommend the best model for production.
```

---

## Resources

### Internal Documentation
- [README.md](README.md) - Project overview
- [PRD: AI-Generated Document Detection](docs/PRD_AI_Generated_Document_Detection.md) - Product requirements
- [Technical Spec: AI Detection](docs/TechSpec_AI_Generated_Document_Detection.md) - Technical architecture
- [Macro Risk Assessment](data/docs/Macro_Risk_Assessment_Dec2025_Validation.md) - Fraud analysis

### External Resources
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [Snowflake SQL Reference](https://docs.snowflake.com/en/sql-reference)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Claude Code Documentation](https://github.com/anthropics/claude-code)

---

## Getting Help

### When to Ask for Help

1. **You're stuck on an error:** Share the full error message with Claude
2. **Model performance is poor:** Ask Claude to diagnose and suggest improvements
3. **Need to understand the data:** Ask for exploratory analysis
4. **Architecture decisions:** Ask Claude to compare approaches and explain tradeoffs
5. **Production readiness:** Ask Claude to identify gaps and risks

### How to Ask Effective Questions

**Good question structure:**
```
[Context] I'm working on the fraud detection model
[Current state] The model has 95% accuracy but only 30% recall
[Goal] I need to improve recall to at least 70%
[Constraints] Without dropping precision below 80%
[Question] What techniques should I try?
```

**What Claude needs to help you:**
- What you're trying to accomplish
- What you've already tried
- Any error messages (full text)
- Relevant data samples or code snippets
- Success criteria

---

## Creating Custom Skills

Claude Code supports creating custom **Skills** - reusable capabilities that can be invoked with specific parameters. Skills are useful for repetitive analysis tasks.

### Skill Directory Structure

```
.claude/
└── skills/
    ├── analyze_fraud_pattern.yaml
    ├── fetch_macro_indicators.yaml
    └── run_model_comparison.yaml
```

### Example Skill: Fraud Pattern Analysis

Create `.claude/skills/analyze_fraud_pattern.yaml`:

```yaml
name: analyze_fraud_pattern
description: Analyze fraud patterns for a specific time period and fraud type
parameters:
  - name: start_date
    type: string
    description: Start date in YYYY-MM-DD format (e.g., 2025-01-01)
    required: true

  - name: end_date
    type: string
    description: End date in YYYY-MM-DD format (e.g., 2025-12-31)
    required: true

  - name: fraud_type
    type: string
    description: Type of fraud (synthetic_id, first_party, third_party_ato, or all)
    required: false
    default: all

instructions: |
  Analyze fraud patterns with the following steps:

  1. Query Snowflake FRAUD_INCIDENTS table:
     - Filter by date range (start_date to end_date)
     - Filter by fraud_type if not 'all'

  2. Calculate key metrics:
     - Total incident count
     - Total loss amount
     - Average loss per incident
     - Detection rate (if detection_method data available)
     - Month-over-month growth rate

  3. Create visualizations:
     - Time series plot of incident counts
     - Loss amount distribution
     - Fraud type breakdown (if fraud_type='all')

  4. Generate summary report with:
     - Key statistics
     - Trend analysis (increasing/decreasing/stable)
     - Notable anomalies or spikes
     - Recommendations for further investigation

  5. Save outputs:
     - CSV: fraud_analysis_{start_date}_{end_date}.csv
     - PNG: fraud_analysis_{start_date}_{end_date}.png
     - MD: fraud_analysis_{start_date}_{end_date}.md (summary report)
```

**Usage:**
```
Invoke the analyze_fraud_pattern skill for January-November 2025,
focusing on synthetic_id fraud
```

### Example Skill: Macro Indicator Dashboard

Create `.claude/skills/fetch_macro_indicators.yaml`:

```yaml
name: fetch_macro_indicators
description: Fetch and visualize key macroeconomic indicators from FRED
parameters:
  - name: months_back
    type: integer
    description: Number of months of historical data to fetch
    required: false
    default: 12

instructions: |
  Fetch and visualize macroeconomic indicators:

  1. Fetch from FRED:
     - UNRATE (Unemployment Rate)
     - FEDFUNDS (Federal Funds Rate)
     - CPIAUCSL (CPI) - convert to YoY % change
     - GDPC1 (Real GDP)

  2. Calculate derived metrics:
     - Inflation rate (YoY CPI change)
     - Rate of change for unemployment
     - Sahm Rule indicator (0.5pp increase check)

  3. Create dashboard with 4 subplots:
     - Unemployment rate trend
     - Federal Funds Rate trend
     - Inflation rate trend
     - GDP growth rate

  4. Add annotations for significant events:
     - Rate hikes/cuts
     - Recession indicators

  5. Save outputs:
     - PNG: macro_dashboard_{current_date}.png
     - CSV: macro_indicators_{current_date}.csv
     - MD: macro_summary_{current_date}.md
```

### Example Skill: Model Comparison

Create `.claude/skills/run_model_comparison.yaml`:

```yaml
name: run_model_comparison
description: Compare multiple ML models for fraud detection
parameters:
  - name: models
    type: array
    description: List of models to compare (logistic_regression, random_forest, xgboost, lightgbm)
    required: false
    default: [random_forest, xgboost]

  - name: test_date_start
    type: string
    description: Start date for test set in YYYY-MM-DD format
    required: false
    default: "2025-11-01"

instructions: |
  Compare ML models for fraud detection:

  1. Load and prepare data:
     - Split into train/test based on test_date_start
     - Extract features using existing pipeline
     - Handle class imbalance

  2. For each model in models list:
     - Train with default hyperparameters
     - Make predictions on test set
     - Calculate metrics: ROC-AUC, Precision, Recall, F1
     - Measure training and inference time

  3. Create comparison table with all metrics

  4. Generate visualizations:
     - ROC curves (all models on same plot)
     - Precision-Recall curves
     - Feature importance comparison
     - Training time comparison bar chart

  5. Provide recommendation:
     - Best model for production (balance accuracy and speed)
     - Trade-offs between models
     - Hyperparameter tuning suggestions

  6. Save outputs:
     - CSV: model_comparison_{current_date}.csv
     - PNG: model_comparison_{current_date}.png
     - MD: model_comparison_report_{current_date}.md
```

### Best Practices for Skills

1. **Be Specific:** Define clear, step-by-step instructions
2. **Include Validation:** Add checks for data quality and errors
3. **Document Outputs:** Specify what files/reports will be generated
4. **Use Defaults:** Provide sensible default parameter values
5. **Error Handling:** Include instructions for common failure cases

### Invoking Skills

Once created, invoke skills naturally:

```
Use the analyze_fraud_pattern skill to analyze synthetic ID fraud
from June to November 2025
```

```
Invoke fetch_macro_indicators for the past 24 months
```

```
Run model_comparison with all available models (logistic_regression,
random_forest, xgboost, lightgbm)
```

---

## Contributing

If you extend this demo with new models or analyses:

1. **Document your work:**
   ```
   Add comments to my new fraud model explaining the approach and rationale
   ```

2. **Create reproducible examples:**
   ```
   Create a Jupyter notebook demonstrating the new customer segmentation model
   ```

3. **Update documentation:**
   ```
   Update the README.md to include documentation for the new churn prediction model
   ```

---

## Feedback

This demo is designed to showcase Claude Code's capabilities for data science and machine learning workflows. If you have suggestions for improvements:

- Open an issue on GitHub
- Share example workflows that would be valuable
- Suggest additional datasets or models to include

---

**Last Updated:** December 14, 2025
**Claude Code Version:** Latest
**Demo Version:** 1.0
