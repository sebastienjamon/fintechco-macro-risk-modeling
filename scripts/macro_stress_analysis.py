"""
Macroeconomic Stress Period Analysis

This script analyzes how payment volumes and fraud rates differ between
macroeconomic stress periods and non-stress periods.

Stress periods are defined using FRED indicators:
- High unemployment (above 75th percentile)
- High interest rates (above 75th percentile)
- Combined stress (both conditions present)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    """
    Load transaction, fraud, and macroeconomic data.
    """
    print("Loading data...")

    # Load transactions
    transactions = pd.read_csv('data/synthetic/payment_transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    # Load fraud data
    fraud_data = pd.read_csv('data/synthetic/fraud_histories.csv')

    # Merge transactions with fraud labels
    data = transactions.merge(fraud_data[['transaction_id', 'is_fraud']],
                              on='transaction_id', how='inner')

    # Load FRED macroeconomic data
    fed_funds = pd.read_csv('data/fred/federal_funds_rate.csv')
    fed_funds['date'] = pd.to_datetime(fed_funds['date'])

    unemployment = pd.read_csv('data/fred/unemployment_rate.csv')
    unemployment['date'] = pd.to_datetime(unemployment['date'])

    cpi = pd.read_csv('data/fred/consumer_price_index.csv')
    cpi['date'] = pd.to_datetime(cpi['date'])

    # Merge FRED data
    macro_data = fed_funds.merge(unemployment, on='date', how='inner')
    macro_data = macro_data.merge(cpi, on='date', how='inner')

    # Calculate year-over-year inflation
    macro_data['inflation_rate_yoy'] = macro_data['cpi_index'].pct_change(12) * 100

    print(f"✓ Loaded {len(data):,} transactions")
    print(f"✓ Loaded {len(macro_data)} months of macro data")

    return data, macro_data


def define_stress_periods(macro_data):
    """
    Define macroeconomic stress periods using thresholds.
    """
    print("\n" + "=" * 70)
    print("Defining Macroeconomic Stress Periods")
    print("=" * 70)

    # Calculate thresholds (75th percentile)
    unemployment_threshold = macro_data['unemployment_rate_percent'].quantile(0.75)
    fed_funds_threshold = macro_data['federal_funds_rate_percent'].quantile(0.75)

    print(f"\nStress Thresholds:")
    print(f"  Unemployment Rate: >{unemployment_threshold:.2f}%")
    print(f"  Federal Funds Rate: >{fed_funds_threshold:.2f}%")

    # Create stress indicators
    macro_data['high_unemployment'] = (
        macro_data['unemployment_rate_percent'] > unemployment_threshold
    ).astype(int)

    macro_data['high_interest_rate'] = (
        macro_data['federal_funds_rate_percent'] > fed_funds_threshold
    ).astype(int)

    # Combined stress: both conditions present
    macro_data['combined_stress'] = (
        (macro_data['high_unemployment'] == 1) &
        (macro_data['high_interest_rate'] == 1)
    ).astype(int)

    # Create a period label
    def get_period_label(row):
        if row['combined_stress'] == 1:
            return 'Combined Stress'
        elif row['high_unemployment'] == 1:
            return 'High Unemployment'
        elif row['high_interest_rate'] == 1:
            return 'High Interest Rate'
        else:
            return 'Normal'

    macro_data['period_type'] = macro_data.apply(get_period_label, axis=1)

    # Summary of stress periods
    print(f"\nPeriod Distribution:")
    period_counts = macro_data['period_type'].value_counts()
    for period, count in period_counts.items():
        print(f"  {period:25s}: {count:2d} months ({count/len(macro_data)*100:.1f}%)")

    # Show date ranges for combined stress periods
    stress_periods = macro_data[macro_data['combined_stress'] == 1]
    if len(stress_periods) > 0:
        print(f"\nCombined Stress Periods:")
        for _, row in stress_periods.iterrows():
            print(f"  {row['date'].strftime('%Y-%m')}: " +
                  f"Unemployment={row['unemployment_rate_percent']:.1f}%, " +
                  f"Fed Funds={row['federal_funds_rate_percent']:.2f}%")

    return macro_data, unemployment_threshold, fed_funds_threshold


def analyze_by_stress_period(transactions, macro_data):
    """
    Analyze transaction volumes and fraud rates by stress period.
    """
    print("\n" + "=" * 70)
    print("Analyzing Payment Volumes and Fraud Rates by Period")
    print("=" * 70)

    # Add year-month to transactions
    transactions['year_month'] = transactions['transaction_date'].dt.to_period('M').dt.to_timestamp()

    # Merge transactions with macro data
    macro_data_for_merge = macro_data.rename(columns={'date': 'year_month'})
    transactions_with_macro = transactions.merge(
        macro_data_for_merge[['year_month', 'period_type', 'high_unemployment',
                               'high_interest_rate', 'combined_stress']],
        on='year_month',
        how='inner'
    )

    # Calculate metrics by period type
    results = []

    for period in ['Normal', 'High Unemployment', 'High Interest Rate', 'Combined Stress']:
        period_data = transactions_with_macro[transactions_with_macro['period_type'] == period]

        if len(period_data) == 0:
            continue

        # Calculate metrics
        total_transactions = len(period_data)
        fraud_count = period_data['is_fraud'].sum()
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0

        # Calculate monthly averages
        months = period_data['year_month'].nunique()
        avg_monthly_volume = total_transactions / months if months > 0 else 0
        avg_transaction_amount = period_data['amount'].mean()
        median_transaction_amount = period_data['amount'].median()

        results.append({
            'Period': period,
            'Total Transactions': total_transactions,
            'Number of Months': months,
            'Avg Monthly Volume': avg_monthly_volume,
            'Total Fraud Cases': fraud_count,
            'Fraud Rate (%)': fraud_rate,
            'Avg Transaction Amount': avg_transaction_amount,
            'Median Transaction Amount': median_transaction_amount
        })

    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "-" * 70)
    print("Summary Statistics by Period Type")
    print("-" * 70)
    print()

    for _, row in results_df.iterrows():
        print(f"{row['Period']}:")
        print(f"  Months in Period: {row['Number of Months']:.0f}")
        print(f"  Total Transactions: {row['Total Transactions']:,.0f}")
        print(f"  Avg Monthly Volume: {row['Avg Monthly Volume']:,.0f} transactions/month")
        print(f"  Fraud Cases: {row['Total Fraud Cases']:.0f}")
        print(f"  Fraud Rate: {row['Fraud Rate (%)']:.3f}%")
        print(f"  Avg Transaction Amount: ${row['Avg Transaction Amount']:,.2f}")
        print(f"  Median Transaction Amount: ${row['Median Transaction Amount']:,.2f}")
        print()

    # Calculate directional comparisons
    print("-" * 70)
    print("Directional Comparisons: Stress vs Normal Periods")
    print("-" * 70)
    print()

    normal_row = results_df[results_df['Period'] == 'Normal'].iloc[0]

    for period in ['High Unemployment', 'High Interest Rate', 'Combined Stress']:
        stress_row = results_df[results_df['Period'] == period]
        if len(stress_row) == 0:
            continue
        stress_row = stress_row.iloc[0]

        volume_change = ((stress_row['Avg Monthly Volume'] - normal_row['Avg Monthly Volume']) /
                        normal_row['Avg Monthly Volume'] * 100)
        fraud_change = stress_row['Fraud Rate (%)'] - normal_row['Fraud Rate (%)']
        fraud_pct_change = ((stress_row['Fraud Rate (%)'] - normal_row['Fraud Rate (%)']) /
                           normal_row['Fraud Rate (%)'] * 100)

        print(f"{period} vs Normal:")
        print(f"  Monthly Transaction Volume: {volume_change:+.1f}%")
        print(f"  Fraud Rate: {fraud_change:+.3f} percentage points ({fraud_pct_change:+.1f}%)")
        print()

    return results_df, transactions_with_macro


def create_visualizations(results_df, transactions_with_macro, macro_data):
    """
    Create visualizations comparing stress and non-stress periods.
    """
    print("=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Monthly transaction volume over time with stress periods shaded
    ax1 = fig.add_subplot(gs[0, :])

    # Aggregate by month
    monthly_volume = transactions_with_macro.groupby('year_month').agg({
        'transaction_id': 'count',
        'combined_stress': 'first'
    }).reset_index()
    monthly_volume.columns = ['year_month', 'transaction_count', 'stress']

    # Plot transaction volume
    ax1.plot(monthly_volume['year_month'], monthly_volume['transaction_count'],
             linewidth=2, color='blue', label='Transaction Volume')

    # Shade stress periods
    stress_periods = monthly_volume[monthly_volume['stress'] == 1]
    for _, row in stress_periods.iterrows():
        ax1.axvspan(row['year_month'], row['year_month'] + pd.DateOffset(months=1),
                   alpha=0.3, color='red')

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Monthly Transaction Count', fontsize=12)
    ax1.set_title('Monthly Transaction Volume (Stress Periods in Red)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Bar chart: Average monthly volume by period
    ax2 = fig.add_subplot(gs[1, 0])

    colors = ['green', 'yellow', 'orange', 'red']
    ax2.bar(range(len(results_df)), results_df['Avg Monthly Volume'], color=colors[:len(results_df)])
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df['Period'], rotation=45, ha='right')
    ax2.set_ylabel('Avg Monthly Volume', fontsize=11)
    ax2.set_title('Average Monthly Transaction Volume', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Bar chart: Fraud rate by period
    ax3 = fig.add_subplot(gs[1, 1])

    ax3.bar(range(len(results_df)), results_df['Fraud Rate (%)'], color=colors[:len(results_df)])
    ax3.set_xticks(range(len(results_df)))
    ax3.set_xticklabels(results_df['Period'], rotation=45, ha='right')
    ax3.set_ylabel('Fraud Rate (%)', fontsize=11)
    ax3.set_title('Fraud Rate by Period', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Bar chart: Average transaction amount by period
    ax4 = fig.add_subplot(gs[1, 2])

    ax4.bar(range(len(results_df)), results_df['Avg Transaction Amount'], color=colors[:len(results_df)])
    ax4.set_xticks(range(len(results_df)))
    ax4.set_xticklabels(results_df['Period'], rotation=45, ha='right')
    ax4.set_ylabel('Avg Amount ($)', fontsize=11)
    ax4.set_title('Average Transaction Amount', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Monthly fraud rate over time
    ax5 = fig.add_subplot(gs[2, 0])

    monthly_fraud = transactions_with_macro.groupby('year_month').agg({
        'is_fraud': 'mean',
        'combined_stress': 'first'
    }).reset_index()
    monthly_fraud.columns = ['year_month', 'fraud_rate', 'stress']
    monthly_fraud['fraud_rate'] = monthly_fraud['fraud_rate'] * 100

    ax5.plot(monthly_fraud['year_month'], monthly_fraud['fraud_rate'],
             linewidth=2, color='red', label='Fraud Rate')

    # Shade stress periods
    stress_periods = monthly_fraud[monthly_fraud['stress'] == 1]
    for _, row in stress_periods.iterrows():
        ax5.axvspan(row['year_month'], row['year_month'] + pd.DateOffset(months=1),
                   alpha=0.3, color='red')

    ax5.set_xlabel('Date', fontsize=12)
    ax5.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax5.set_title('Monthly Fraud Rate (Stress Periods Shaded)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Macro indicators over time
    ax6 = fig.add_subplot(gs[2, 1])

    ax6.plot(macro_data['date'], macro_data['unemployment_rate_percent'],
             linewidth=2, label='Unemployment Rate', color='orange')
    ax6.set_xlabel('Date', fontsize=12)
    ax6.set_ylabel('Unemployment Rate (%)', fontsize=12, color='orange')
    ax6.tick_params(axis='y', labelcolor='orange')

    ax6_twin = ax6.twinx()
    ax6_twin.plot(macro_data['date'], macro_data['federal_funds_rate_percent'],
                  linewidth=2, label='Fed Funds Rate', color='blue', linestyle='--')
    ax6_twin.set_ylabel('Fed Funds Rate (%)', fontsize=12, color='blue')
    ax6_twin.tick_params(axis='y', labelcolor='blue')

    ax6.set_title('Macroeconomic Indicators', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Box plot: Transaction amounts by period
    ax7 = fig.add_subplot(gs[2, 2])

    period_order = ['Normal', 'High Unemployment', 'High Interest Rate', 'Combined Stress']
    period_data_for_box = [
        transactions_with_macro[transactions_with_macro['period_type'] == p]['amount'].values
        for p in period_order if p in transactions_with_macro['period_type'].unique()
    ]
    period_labels_for_box = [
        p for p in period_order if p in transactions_with_macro['period_type'].unique()
    ]

    bp = ax7.boxplot(period_data_for_box, labels=period_labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax7.set_xticklabels(period_labels_for_box, rotation=45, ha='right')
    ax7.set_ylabel('Transaction Amount ($)', fontsize=11)
    ax7.set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
    ax7.set_ylim(0, 500)  # Focus on typical range
    ax7.grid(True, alpha=0.3, axis='y')

    plt.savefig('data/fred/macro_stress_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to: data/fred/macro_stress_analysis.png")

    plt.close()


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 70)
    print("MACROECONOMIC STRESS PERIOD ANALYSIS")
    print("FinTechCo Payment Volumes and Fraud Rates")
    print("=" * 70)
    print()

    # Load data
    transactions, macro_data = load_data()

    # Define stress periods
    macro_data, unemployment_threshold, fed_funds_threshold = define_stress_periods(macro_data)

    # Analyze by stress period
    results_df, transactions_with_macro = analyze_by_stress_period(transactions, macro_data)

    # Save results
    results_df.to_csv('data/fred/stress_period_analysis.csv', index=False)
    print("\n✓ Results saved to: data/fred/stress_period_analysis.csv")

    # Create visualizations
    create_visualizations(results_df, transactions_with_macro, macro_data)

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    print("This analysis compared payment behavior during macroeconomic stress vs")
    print("normal periods, using elevated unemployment (>4.2%) and interest rates")
    print("(>5.13%) as stress indicators.")
    print()
    print("Directional Observations:")
    print()
    print("1. TRANSACTION VOLUME")
    print("   During stress periods, transaction volumes show directional changes")
    print("   compared to normal periods. Review the visualizations to observe")
    print("   whether volumes increase, decrease, or remain stable.")
    print()
    print("2. FRAUD RATES")
    print("   Fraud rates may vary between stress and non-stress periods. The")
    print("   analysis shows the fraud rate differential across period types.")
    print()
    print("3. TRANSACTION AMOUNTS")
    print("   Average transaction amounts provide insight into customer behavior")
    print("   changes during economic stress.")
    print()
    print("Note: This is a descriptive analysis showing directional relationships.")
    print("Statistical significance testing would be needed for formal inference.")
    print()


if __name__ == "__main__":
    main()
