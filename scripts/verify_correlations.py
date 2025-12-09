"""
Quick verification script to check if the synthetic data has the expected correlations.
"""

import pandas as pd
import numpy as np

# Load data
print("Loading data...")
transactions = pd.read_csv('data/synthetic/payment_transactions.csv')
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

fraud_data = pd.read_csv('data/synthetic/fraud_histories.csv')
unemployment = pd.read_csv('data/fred/unemployment_rate.csv')
unemployment['date'] = pd.to_datetime(unemployment['date'])
fed_funds = pd.read_csv('data/fred/federal_funds_rate.csv')
fed_funds['date'] = pd.to_datetime(fed_funds['date'])

# Merge transactions with fraud
data = transactions.merge(fraud_data, on='transaction_id', how='inner')

# Create monthly aggregations
data['year_month'] = data['transaction_date'].dt.to_period('M').dt.to_timestamp()

monthly_tx = data.groupby('year_month').agg({
    'transaction_id': 'count',
    'is_fraud': 'mean',
}).reset_index()

monthly_tx.columns = ['year_month', 'tx_volume', 'fraud_rate']
monthly_tx['fraud_rate_pct'] = monthly_tx['fraud_rate'] * 100

# Merge with macro data
macro = unemployment.merge(fed_funds, left_on='date', right_on='date', how='inner')
macro = macro.rename(columns={'date': 'year_month'})

monthly_combined = monthly_tx.merge(macro, on='year_month', how='inner')

# Calculate correlations
corr_unemp_fraud = monthly_combined['unemployment_rate_percent'].corr(monthly_combined['fraud_rate_pct'])
corr_unemp_vol = monthly_combined['unemployment_rate_percent'].corr(monthly_combined['tx_volume'])
corr_rate_vol = monthly_combined['federal_funds_rate_percent'].corr(monthly_combined['tx_volume'])

print("\n" + "="*70)
print("CORRELATION VERIFICATION")
print("="*70)

print("\nH1: Fraud ↑ with Unemployment ↑")
print(f"   Correlation: {corr_unemp_fraud:.3f}")
print(f"   Expected: POSITIVE (>0)")
print(f"   Status: {'✓ CONFIRMS' if corr_unemp_fraud > 0 else '✗ CONTRADICTS'}")

print("\nH2: Volume ↓ with Unemployment ↑")
print(f"   Correlation: {corr_unemp_vol:.3f}")
print(f"   Expected: NEGATIVE (<0)")
print(f"   Status: {'✓ CONFIRMS' if corr_unemp_vol < 0 else '✗ CONTRADICTS'}")

print("\nH2: Volume ↓ with Interest Rates ↑")
print(f"   Correlation: {corr_rate_vol:.3f}")
print(f"   Expected: NEGATIVE (<0)")
print(f"   Status: {'✓ CONFIRMS' if corr_rate_vol < 0 else '✗ CONTRADICTS'}")

# Additional analysis
unemp_median = monthly_combined['unemployment_rate_percent'].median()
high_unemp = monthly_combined[monthly_combined['unemployment_rate_percent'] > unemp_median]
low_unemp = monthly_combined[monthly_combined['unemployment_rate_percent'] <= unemp_median]

fraud_diff_pct = ((high_unemp['fraud_rate_pct'].mean() / low_unemp['fraud_rate_pct'].mean()) - 1) * 100
vol_diff_pct = ((high_unemp['tx_volume'].mean() / low_unemp['tx_volume'].mean()) - 1) * 100

print("\n" + "="*70)
print("EFFECT SIZES")
print("="*70)

print(f"\nHigh unemployment (>{unemp_median:.1f}%) vs Low (≤{unemp_median:.1f}%):")
print(f"   Fraud rate change: {fraud_diff_pct:+.1f}%")
print(f"   Volume change: {vol_diff_pct:+.1f}%")

print("\n" + "="*70)

# Create summary table like in the notebook
validation_summary = pd.DataFrame({
    'Hypothesis': [
        'H1: Fraud ↑ with Unemployment ↑',
        'H2: Volume ↓ with Unemployment ↑',
        'H2: Volume ↓ with Interest Rates ↑'
    ],
    'Correlation': [
        f"{corr_unemp_fraud:.3f}",
        f"{corr_unemp_vol:.3f}",
        f"{corr_rate_vol:.3f}"
    ],
    'Effect Size': [
        f"{fraud_diff_pct:+.1f}%",
        f"{vol_diff_pct:+.1f}%",
        "N/A"
    ],
    'Direction': [
        '✓ Confirms' if corr_unemp_fraud > 0 else '✗ Contradicts',
        '✓ Confirms' if vol_diff_pct < 0 else '✗ Contradicts',
        '✓ Confirms' if corr_rate_vol < 0 else '✗ Contradicts'
    ]
})

print("\nVALIDATION SUMMARY TABLE")
print("="*70)
print(validation_summary.to_string(index=False))
print("="*70)
