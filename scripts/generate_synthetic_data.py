"""
Generate synthetic data for FinTechCo demonstration environment.
This script creates realistic payment transactions, fraud histories, and internal metrics.
Data is generated with correlations to macroeconomic indicators (unemployment, interest rates).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def load_macro_data():
    """
    Load FRED macroeconomic data to incorporate correlations.
    Returns unemployment and interest rate data by month.
    """
    try:
        unemployment = pd.read_csv('data/fred/unemployment_rate.csv')
        unemployment['date'] = pd.to_datetime(unemployment['date'])
        unemployment['year_month'] = unemployment['date'].dt.to_period('M')

        fed_funds = pd.read_csv('data/fred/federal_funds_rate.csv')
        fed_funds['date'] = pd.to_datetime(fed_funds['date'])
        fed_funds['year_month'] = fed_funds['date'].dt.to_period('M')

        # Merge
        macro_data = unemployment[['year_month', 'unemployment_rate_percent']].merge(
            fed_funds[['year_month', 'federal_funds_rate_percent']],
            on='year_month',
            how='inner'
        )

        # Normalize to create impact multipliers
        # Higher unemployment = higher fraud, lower volume
        # Higher interest rates = lower volume
        macro_data['unemployment_normalized'] = (
            macro_data['unemployment_rate_percent'] - macro_data['unemployment_rate_percent'].min()
        ) / (macro_data['unemployment_rate_percent'].max() - macro_data['unemployment_rate_percent'].min())

        macro_data['fed_funds_normalized'] = (
            macro_data['federal_funds_rate_percent'] - macro_data['federal_funds_rate_percent'].min()
        ) / (macro_data['federal_funds_rate_percent'].max() - macro_data['federal_funds_rate_percent'].min())

        return macro_data
    except FileNotFoundError:
        print("Warning: FRED data not found. Generating without macro correlations.")
        return None


def generate_payment_transactions(n_transactions=50000, start_date='2022-01-01', end_date='2025-11-30', macro_data=None):
    """
    Generate realistic synthetic payment transaction data.
    Incorporates macro effects: lower volume during high unemployment/interest rates.
    """
    print(f"Generating {n_transactions} payment transactions...")

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days

    # Generate transaction dates with macro-adjusted distribution
    # If we have macro data, distribute transactions based on economic conditions
    if macro_data is not None:
        # Create daily macro indicators by forward-filling monthly data
        date_index = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_macro = pd.DataFrame({'date': date_index})
        daily_macro['year_month'] = daily_macro['date'].dt.to_period('M')
        daily_macro = daily_macro.merge(macro_data, on='year_month', how='left')
        daily_macro = daily_macro.ffill().bfill()

        # Calculate volume adjustment: lower during high unemployment/rates
        # Combined effect: -30% volume at highest stress, +0% at lowest
        daily_macro['volume_multiplier'] = 1.0 - (
            0.20 * daily_macro['unemployment_normalized'] +  # 20% effect from unemployment
            0.10 * daily_macro['fed_funds_normalized']  # 10% effect from interest rates
        )

        # Generate dates weighted by volume multiplier
        date_weights = daily_macro['volume_multiplier'].values
        date_weights = date_weights / date_weights.sum()  # Normalize to probabilities

        # Sample dates according to macro-adjusted probabilities
        date_indices = np.random.choice(len(daily_macro), size=n_transactions, p=date_weights)
        dates = [daily_macro.iloc[idx]['date'].to_pydatetime() for idx in date_indices]
    else:
        # Fallback: generate dates with triangular distribution (more recent transactions)
        dates = [start + timedelta(days=int(random.triangular(0, date_range, date_range * 0.8)))
                 for _ in range(n_transactions)]

    # Transaction types and their typical amounts
    transaction_types = ['purchase', 'withdrawal', 'transfer', 'payment', 'refund']
    type_weights = [0.50, 0.15, 0.20, 0.10, 0.05]

    # Merchant categories
    merchant_categories = [
        'retail', 'groceries', 'restaurants', 'entertainment', 'travel',
        'utilities', 'healthcare', 'education', 'online_services', 'other'
    ]

    # Generate realistic customer IDs (10000-99999)
    customer_ids = np.random.randint(10000, 100000, n_transactions)

    # Generate transaction amounts based on type
    amounts = []
    types = []
    categories = []

    for _ in range(n_transactions):
        trans_type = random.choices(transaction_types, weights=type_weights)[0]
        types.append(trans_type)

        # Different amount distributions for different transaction types
        if trans_type == 'purchase':
            amount = abs(np.random.lognormal(3.5, 1.2))  # Mean ~$50, right-skewed
            categories.append(random.choice(merchant_categories))
        elif trans_type == 'withdrawal':
            amount = random.choice([20, 40, 60, 80, 100, 200])  # Common ATM amounts
            categories.append('atm')
        elif trans_type == 'transfer':
            amount = abs(np.random.lognormal(5.0, 1.5))  # Mean ~$200
            categories.append('p2p_transfer')
        elif trans_type == 'payment':
            amount = abs(np.random.lognormal(4.5, 1.0))  # Mean ~$100
            categories.append(random.choice(['bill_payment', 'subscription', 'loan_payment']))
        else:  # refund
            amount = abs(np.random.lognormal(3.0, 1.0))  # Mean ~$25
            categories.append('refund')

        amounts.append(round(amount, 2))

    # Generate payment methods
    payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet', 'cash']
    method_weights = [0.40, 0.30, 0.15, 0.10, 0.05]
    methods = random.choices(payment_methods, weights=method_weights, k=n_transactions)

    # Generate locations (US cities)
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
              'Austin', 'Jacksonville', 'Seattle', 'Denver', 'Boston']
    locations = [random.choice(cities) for _ in range(n_transactions)]

    # Generate device types
    devices = ['mobile', 'desktop', 'tablet', 'pos_terminal', 'atm']
    device_weights = [0.45, 0.25, 0.10, 0.15, 0.05]
    device_types = random.choices(devices, weights=device_weights, k=n_transactions)

    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': [f'TXN{str(i).zfill(8)}' for i in range(1, n_transactions + 1)],
        'customer_id': customer_ids,
        'transaction_date': dates,
        'transaction_type': types,
        'amount': amounts,
        'merchant_category': categories,
        'payment_method': methods,
        'location': locations,
        'device_type': device_types,
        'status': ['completed'] * n_transactions  # Most transactions complete successfully
    })

    # Add some failed transactions (2%)
    failed_indices = np.random.choice(df.index, size=int(n_transactions * 0.02), replace=False)
    df.loc[failed_indices, 'status'] = 'failed'

    # Sort by date
    df = df.sort_values('transaction_date').reset_index(drop=True)

    return df


def generate_fraud_data(transactions_df, base_fraud_rate=0.015, macro_data=None):
    """
    Generate fraud labels and fraud-related features for transactions.
    Incorporates macro effects: higher fraud during high unemployment.
    """
    print(f"Generating fraud data with base {base_fraud_rate*100}% fraud rate...")

    # If we have macro data, adjust fraud probability by unemployment
    if macro_data is not None:
        # Merge transactions with macro data by month
        trans_with_macro = transactions_df.copy()
        trans_with_macro['year_month'] = pd.to_datetime(trans_with_macro['transaction_date']).dt.to_period('M')
        trans_with_macro = trans_with_macro.merge(
            macro_data[['year_month', 'unemployment_normalized']],
            on='year_month',
            how='left'
        )

        # Fill any missing values with mean unemployment
        trans_with_macro['unemployment_normalized'] = trans_with_macro['unemployment_normalized'].fillna(
            trans_with_macro['unemployment_normalized'].mean()
        )

        # Fraud probability increases with unemployment
        # +50% fraud rate at highest unemployment vs lowest
        trans_with_macro['fraud_probability'] = base_fraud_rate * (1.0 + 0.5 * trans_with_macro['unemployment_normalized'])

        # Generate fraud labels based on adjusted probabilities
        fraud_labels = np.random.binomial(1, trans_with_macro['fraud_probability'].values)
        n_frauds = fraud_labels.sum()
    else:
        # Fallback: fixed fraud rate
        n_frauds = int(len(transactions_df) * base_fraud_rate)
        fraud_labels = np.zeros(len(transactions_df))
        fraud_indices = np.random.choice(len(transactions_df), size=n_frauds, replace=False)
        fraud_labels[fraud_indices] = 1

    fraud_indices = np.where(fraud_labels == 1)[0]

    # Fraud characteristics
    fraud_scores = np.random.uniform(0, 0.3, len(transactions_df))  # Normal transactions: low score
    fraud_scores[fraud_indices] = np.random.uniform(0.6, 1.0, n_frauds)  # Fraudulent: high score

    # Fraud types (only for fraudulent transactions)
    fraud_types = [''] * len(transactions_df)
    fraud_type_options = ['stolen_card', 'account_takeover', 'synthetic_identity',
                          'chargeback_fraud', 'friendly_fraud']
    for idx in fraud_indices:
        fraud_types[idx] = random.choice(fraud_type_options)

    # Detection methods
    detection_methods = [''] * len(transactions_df)
    detection_options = ['ml_model', 'rule_based', 'manual_review', 'customer_report']
    for idx in fraud_indices:
        detection_methods[idx] = random.choice(detection_options)

    # Time to detection (hours) - only for fraud cases
    time_to_detection = [np.nan] * len(transactions_df)
    for idx in fraud_indices:
        time_to_detection[idx] = abs(np.random.exponential(24))  # Mean 24 hours

    # Create fraud DataFrame
    fraud_df = pd.DataFrame({
        'transaction_id': transactions_df['transaction_id'],
        'is_fraud': fraud_labels.astype(int),
        'fraud_score': fraud_scores.round(3),
        'fraud_type': fraud_types,
        'detection_method': detection_methods,
        'time_to_detection_hours': time_to_detection
    })

    return fraud_df


def generate_customer_metrics(transactions_df, start_date='2022-01-01', end_date='2025-11-30'):
    """
    Generate aggregated customer-level metrics over time.
    """
    print("Generating customer metrics...")

    # Create monthly aggregations
    transactions_df['year_month'] = pd.to_datetime(transactions_df['transaction_date']).dt.to_period('M')

    customer_metrics = transactions_df.groupby(['year_month', 'customer_id']).agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'std']
    }).reset_index()

    # Flatten column names
    customer_metrics.columns = ['year_month', 'customer_id', 'transaction_count',
                                 'total_spending', 'avg_transaction_amount', 'spending_volatility']

    customer_metrics['year_month'] = customer_metrics['year_month'].dt.to_timestamp()

    return customer_metrics


def generate_internal_metrics(start_date='2022-01-01', end_date='2025-11-30'):
    """
    Generate company-level internal metrics (daily/monthly).
    """
    print("Generating internal company metrics...")

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)

    # Base values with trends
    base_daily_volume = 1000
    base_revenue = 50000

    # Add trends and seasonality
    trend = np.linspace(0, 0.3, n_days)  # 30% growth over period
    seasonality = 0.1 * np.sin(np.arange(n_days) * 2 * np.pi / 365)  # Annual seasonality
    noise = np.random.normal(0, 0.05, n_days)

    daily_transaction_volume = base_daily_volume * (1 + trend + seasonality + noise)
    daily_revenue = base_revenue * (1 + trend + seasonality + noise)

    # Customer acquisition
    daily_new_customers = np.random.poisson(50, n_days) + trend * 20

    # Customer churn rate (%)
    base_churn = 2.5
    churn_rate = base_churn + np.random.normal(0, 0.3, n_days)
    churn_rate = np.clip(churn_rate, 1.0, 5.0)

    # System metrics
    avg_response_time_ms = 150 + np.random.normal(0, 20, n_days)
    system_uptime_pct = 99.5 + np.random.normal(0, 0.3, n_days)
    system_uptime_pct = np.clip(system_uptime_pct, 97, 100)

    # Fraud metrics
    fraud_detection_rate = 85 + np.random.normal(0, 5, n_days)
    fraud_detection_rate = np.clip(fraud_detection_rate, 70, 98)

    false_positive_rate = 3 + np.random.normal(0, 0.5, n_days)
    false_positive_rate = np.clip(false_positive_rate, 1, 8)

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'date': date_range,
        'daily_transaction_volume': daily_transaction_volume.astype(int),
        'daily_revenue_usd': daily_revenue.round(2),
        'daily_new_customers': daily_new_customers.astype(int),
        'customer_churn_rate_pct': churn_rate.round(2),
        'avg_response_time_ms': avg_response_time_ms.round(1),
        'system_uptime_pct': system_uptime_pct.round(2),
        'fraud_detection_rate_pct': fraud_detection_rate.round(2),
        'false_positive_rate_pct': false_positive_rate.round(2)
    })

    # Add monthly metrics
    metrics_df['year_month'] = metrics_df['date'].dt.to_period('M')
    monthly_metrics = metrics_df.groupby('year_month').agg({
        'daily_transaction_volume': 'sum',
        'daily_revenue_usd': 'sum',
        'daily_new_customers': 'sum',
        'customer_churn_rate_pct': 'mean',
        'avg_response_time_ms': 'mean',
        'system_uptime_pct': 'mean',
        'fraud_detection_rate_pct': 'mean',
        'false_positive_rate_pct': 'mean'
    }).reset_index()

    monthly_metrics.columns = ['year_month', 'monthly_transaction_volume', 'monthly_revenue_usd',
                               'monthly_new_customers', 'avg_churn_rate_pct',
                               'avg_response_time_ms', 'avg_uptime_pct',
                               'avg_fraud_detection_rate_pct', 'avg_false_positive_rate_pct']

    monthly_metrics['year_month'] = monthly_metrics['year_month'].dt.to_timestamp()

    return metrics_df, monthly_metrics


def main():
    """
    Generate all synthetic data and save to CSV files.
    Data incorporates correlations with macroeconomic indicators.
    """
    print("=" * 60)
    print("FinTechCo Synthetic Data Generation")
    print("With Macroeconomic Correlations")
    print("=" * 60)
    print()

    # Load macroeconomic data
    print("Loading macroeconomic data...")
    macro_data = load_macro_data()
    if macro_data is not None:
        print(f"✓ Loaded macro data: {len(macro_data)} months")
        print(f"  Unemployment range: {macro_data['unemployment_rate_percent'].min():.1f}% - {macro_data['unemployment_rate_percent'].max():.1f}%")
        print(f"  Interest rate range: {macro_data['federal_funds_rate_percent'].min():.2f}% - {macro_data['federal_funds_rate_percent'].max():.2f}%")
        print(f"  Correlations will be introduced:")
        print(f"    • H1: Fraud ↑ with Unemployment ↑ (+50% at max unemployment)")
        print(f"    • H2: Volume ↓ with Unemployment ↑ (-20% at max unemployment)")
        print(f"    • H2: Volume ↓ with Interest Rates ↑ (-10% at max rates)")
    else:
        print("⚠ No macro data found - generating without correlations")
    print()

    # Generate payment transactions
    transactions_df = generate_payment_transactions(n_transactions=50000, macro_data=macro_data)
    transactions_df.to_csv('data/synthetic/payment_transactions.csv', index=False)
    print(f"✓ Saved payment_transactions.csv ({len(transactions_df)} records)")
    print()

    # Generate fraud data
    fraud_df = generate_fraud_data(transactions_df, macro_data=macro_data)
    fraud_df.to_csv('data/synthetic/fraud_histories.csv', index=False)
    print(f"✓ Saved fraud_histories.csv ({len(fraud_df)} records, {fraud_df['is_fraud'].sum()} frauds)")
    print()

    # Generate customer metrics
    customer_metrics_df = generate_customer_metrics(transactions_df)
    customer_metrics_df.to_csv('data/synthetic/customer_metrics.csv', index=False)
    print(f"✓ Saved customer_metrics.csv ({len(customer_metrics_df)} records)")
    print()

    # Generate internal metrics
    daily_metrics_df, monthly_metrics_df = generate_internal_metrics()
    daily_metrics_df.to_csv('data/synthetic/daily_internal_metrics.csv', index=False)
    monthly_metrics_df.to_csv('data/synthetic/monthly_internal_metrics.csv', index=False)
    print(f"✓ Saved daily_internal_metrics.csv ({len(daily_metrics_df)} records)")
    print(f"✓ Saved monthly_internal_metrics.csv ({len(monthly_metrics_df)} records)")
    print()

    print("=" * 60)
    print("Data generation complete!")
    print("=" * 60)

    # Display sample statistics
    print("\nSample Statistics:")
    print(f"  Total Transactions: {len(transactions_df):,}")
    print(f"  Date Range: {transactions_df['transaction_date'].min().date()} to {transactions_df['transaction_date'].max().date()}")
    print(f"  Total Transaction Value: ${transactions_df['amount'].sum():,.2f}")
    print(f"  Fraud Cases: {fraud_df['is_fraud'].sum():,} ({fraud_df['is_fraud'].mean()*100:.2f}%)")
    print(f"  Unique Customers: {transactions_df['customer_id'].nunique():,}")


if __name__ == "__main__":
    main()
