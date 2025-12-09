# FinTechCo Macro Risk Modeling - Data Science Demonstration Environment

A comprehensive demonstration environment for Data Scientists at FinTechCo, featuring realistic synthetic internal data, real-world FRED macroeconomic indicators, and working machine learning models.

## Overview

This project provides a complete demonstration environment that allows Data Scientists to:

- Work with realistic synthetic payment transaction data
- Analyze real macroeconomic trends using FRED data
- Build and test predictive models combining internal and external data
- Understand the relationship between macro economic indicators and business performance
- Develop fraud detection systems using machine learning

## Project Structure

```
fintechco-macro-risk-modeling/
├── data/
│   ├── synthetic/          # Synthetic internal company data
│   │   ├── payment_transactions.csv
│   │   ├── fraud_histories.csv
│   │   ├── customer_metrics.csv
│   │   ├── daily_internal_metrics.csv
│   │   ├── monthly_internal_metrics.csv
│   │   ├── fraud_predictions.csv
│   │   └── fraud_detection_results.png
│   └── fred/              # Real FRED macroeconomic data
│       ├── federal_funds_rate.csv
│       ├── consumer_price_index.csv
│       ├── unemployment_rate.csv
│       ├── real_gdp.csv
│       ├── revenue_predictions.csv
│       └── linear_regression_results.png
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── fetch_fred_data.py
│   ├── linear_regression_model.py
│   └── fraud_classification_model.py
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data Description

### Synthetic Internal Data

#### 1. Payment Transactions (50,000 records)
**File:** `data/synthetic/payment_transactions.csv`

Realistic payment transaction data with the following features:
- Transaction IDs, customer IDs, dates
- Transaction types: purchase, withdrawal, transfer, payment, refund
- Amount, merchant category, payment method
- Location, device type, transaction status
- Date range: 2022-01-01 to 2025-11-30

#### 2. Fraud Histories (50,000 records)
**File:** `data/synthetic/fraud_histories.csv`

Fraud labels and detection metadata:
- Fraud indicator (1.5% fraud rate)
- Fraud risk scores (0-1 scale)
- Fraud types: stolen_card, account_takeover, synthetic_identity, etc.
- Detection methods: ml_model, rule_based, manual_review, customer_report
- Time to detection metrics

#### 3. Customer Metrics (Monthly aggregations)
**File:** `data/synthetic/customer_metrics.csv`

Customer-level behavioral metrics:
- Transaction counts per customer per month
- Total spending, average transaction amount
- Spending volatility (standard deviation)

#### 4. Internal Company Metrics
**Files:**
- `data/synthetic/daily_internal_metrics.csv` (1,430 records)
- `data/synthetic/monthly_internal_metrics.csv` (47 records)

Business performance indicators:
- Daily/Monthly transaction volumes
- Revenue (USD)
- Customer acquisition and churn rates
- System performance (response time, uptime)
- Fraud detection rates and false positive rates

### Real FRED Macroeconomic Data

All FRED data covers the period from 2022-01-01 to 2025-11-30 (or latest available).

#### 1. Federal Funds Rate
**File:** `data/fred/federal_funds_rate.csv`
- Source: FRED Series ID: FEDFUNDS
- Frequency: Monthly
- The interest rate at which depository institutions trade federal funds

#### 2. Consumer Price Index (CPI)
**File:** `data/fred/consumer_price_index.csv`
- Source: FRED Series ID: CPIAUCSL
- Frequency: Monthly
- Index 1982-1984=100
- Key indicator of inflation

#### 3. Unemployment Rate
**File:** `data/fred/unemployment_rate.csv`
- Source: FRED Series ID: UNRATE
- Frequency: Monthly
- Percentage of unemployed in the labor force

#### 4. Real GDP
**File:** `data/fred/real_gdp.csv`
- Source: FRED Series ID: GDPC1
- Frequency: Quarterly
- Billions of Chained 2017 Dollars

## Machine Learning Models

### 1. Revenue Prediction Model (Linear Regression)

**Script:** `scripts/linear_regression_model.py`

**Purpose:** Predict FinTechCo's monthly revenue using macroeconomic indicators

**Features:**
- Federal funds rate
- Unemployment rate
- Year-over-year inflation rate
- Monthly transaction volume
- Customer churn rate

**Model Performance:**
- Training R² Score: 1.0000
- Test R² Score: 1.0000
- Test RMSE: ~$95
- Test MAE: ~$76

**Key Insights:**
- Higher federal funds rates correlate with higher revenue
- Higher transaction volume drives revenue growth
- Higher unemployment and churn rates negatively impact revenue

**Outputs:**
- `data/fred/revenue_predictions.csv` - Predictions vs actuals
- `data/fred/linear_regression_results.png` - Visualizations

**Usage:**
```bash
python3 scripts/linear_regression_model.py
```

### 2. Fraud Detection Model (Random Forest Classifier)

**Script:** `scripts/fraud_classification_model.py`

**Purpose:** Classify transactions as fraudulent or legitimate

**Features:**
- Transaction amount and log-transformed amount
- Transaction type, merchant category, payment method
- Location, device type, transaction status
- Time-based features: hour, day of week, weekend, night
- Customer behavior: average amount, transaction count, amount deviation

**Model Performance:**
- Training ROC-AUC: 0.9943
- Test ROC-AUC: 0.4583
- The model shows signs of overfitting (common with highly imbalanced data)
- In production, consider using SMOTE, adjusting class weights, or ensemble methods

**Top Important Features:**
1. Amount deviation from customer average
2. Customer average amount
3. Log-transformed amount
4. Transaction amount
5. Location

**Outputs:**
- `data/synthetic/fraud_predictions.csv` - Predictions with probabilities
- `data/synthetic/fraud_detection_results.png` - Visualizations

**Usage:**
```bash
python3 scripts/fraud_classification_model.py
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone or navigate to this repository:
```bash
cd fintechco-macro-risk-modeling
```

2. Install required Python packages:
```bash
pip3 install -r requirements.txt
```

### Generate Data

1. Generate synthetic internal data:
```bash
python3 scripts/generate_synthetic_data.py
```

2. Fetch real FRED macroeconomic data:
```bash
python3 scripts/fetch_fred_data.py
```

### Run Models

1. Run the revenue prediction model:
```bash
python3 scripts/linear_regression_model.py
```

2. Run the fraud detection model:
```bash
python3 scripts/fraud_classification_model.py
```

## Use Cases

### 1. Strategic Planning
- Use the revenue prediction model to forecast revenue under different macroeconomic scenarios
- Understand how interest rate changes might impact business performance
- Plan for economic downturns by analyzing unemployment correlations

### 2. Risk Assessment
- Identify periods of elevated fraud risk
- Analyze fraud patterns across different transaction types and customer segments
- Optimize fraud detection thresholds to balance false positives and false negatives

### 3. Research & Development
- Test new feature engineering approaches for fraud detection
- Experiment with different model architectures
- Develop custom risk scoring models
- Create ensemble models combining multiple data sources

### 4. Educational Purposes
- Learn about time series analysis with economic data
- Practice feature engineering for fraud detection
- Understand model evaluation metrics for imbalanced classification
- Explore the relationship between macro trends and business metrics

## Model Improvements & Next Steps

### Revenue Prediction Model
- Add more macroeconomic indicators (consumer confidence, housing starts)
- Incorporate lagged features to capture delayed economic effects
- Test time series models (ARIMA, Prophet) for better temporal patterns
- Add seasonality adjustments

### Fraud Detection Model
- Address class imbalance with SMOTE or undersampling
- Add more behavioral features (velocity checks, device fingerprinting)
- Implement online learning for real-time adaptation
- Create ensemble models combining multiple algorithms
- Add explainability features (SHAP values) for regulatory compliance

## Data Dictionary

### Payment Transactions
| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string | Unique transaction identifier |
| customer_id | int | Customer identifier |
| transaction_date | datetime | Date and time of transaction |
| transaction_type | string | Type: purchase, withdrawal, transfer, payment, refund |
| amount | float | Transaction amount in USD |
| merchant_category | string | Category of merchant or transaction |
| payment_method | string | Method: credit_card, debit_card, bank_transfer, etc. |
| location | string | City where transaction occurred |
| device_type | string | Device: mobile, desktop, tablet, pos_terminal, atm |
| status | string | Status: completed, failed |

### Fraud Histories
| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string | Unique transaction identifier |
| is_fraud | int | 1 if fraudulent, 0 if legitimate |
| fraud_score | float | ML model fraud risk score (0-1) |
| fraud_type | string | Type of fraud if applicable |
| detection_method | string | How fraud was detected |
| time_to_detection_hours | float | Hours until fraud was detected |

### FRED Data
| Series | Column | Description |
|--------|--------|-------------|
| Federal Funds Rate | federal_funds_rate_percent | Interest rate (%) |
| CPI | cpi_index | Consumer price index value |
| Unemployment | unemployment_rate_percent | Unemployment rate (%) |
| Real GDP | real_gdp_billions | GDP in billions of 2017 dollars |

## Contributing

This is a demonstration environment. Feel free to:
- Extend the synthetic data generation with more realistic patterns
- Add new macroeconomic indicators from FRED
- Implement additional models (time series, deep learning, etc.)
- Create Jupyter notebooks for exploratory analysis
- Improve model performance and interpretability

## License

This project is for educational and demonstration purposes.

## Acknowledgments

- **FRED (Federal Reserve Economic Data)** - Real macroeconomic data
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **matplotlib/seaborn** - Visualizations

## Contact

For questions or feedback about this demonstration environment, please contact the FinTechCo Data Science team.

---

**Last Updated:** December 2025
