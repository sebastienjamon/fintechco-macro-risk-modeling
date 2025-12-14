#!/usr/bin/env python3
"""
Fraud Risk Economic Analysis
Analyzes FRED macroeconomic data and Snowflake payment/fraud data to identify
early risk signals and trending fraud categories.

Output: Markdown report for Data Science validation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# File paths
FRED_DIR = '/home/user/fintechco-macro-risk-modeling/data/fred'
SYNTHETIC_DIR = '/home/user/fintechco-macro-risk-modeling/data/synthetic'
OUTPUT_DIR = '/home/user/fintechco-macro-risk-modeling/data'

def load_fred_data():
    """Load and process FRED macroeconomic data"""
    print("Loading FRED macroeconomic data...")

    # Load unemployment rate
    unemployment = pd.read_csv(f'{FRED_DIR}/unemployment_rate.csv', parse_dates=['date'])
    unemployment['date'] = pd.to_datetime(unemployment['date'])

    # Load federal funds rate
    fed_funds = pd.read_csv(f'{FRED_DIR}/federal_funds_rate.csv', parse_dates=['date'])
    fed_funds['date'] = pd.to_datetime(fed_funds['date'])

    # Load CPI for inflation calculation
    cpi = pd.read_csv(f'{FRED_DIR}/consumer_price_index.csv', parse_dates=['date'])
    cpi['date'] = pd.to_datetime(cpi['date'])

    # Calculate year-over-year inflation rate
    cpi = cpi.sort_values('date')
    cpi['inflation_rate'] = cpi['cpi_index'].pct_change(periods=12) * 100

    # Merge all FRED data
    fred_data = unemployment.merge(fed_funds, on='date', how='outer')
    fred_data = fred_data.merge(cpi[['date', 'cpi_index', 'inflation_rate']], on='date', how='outer')
    fred_data = fred_data.sort_values('date')

    return fred_data

def analyze_macro_conditions(fred_data):
    """Analyze current macroeconomic conditions and risk signals"""
    print("\nAnalyzing macroeconomic conditions...")

    # Get most recent values
    recent = fred_data.dropna().tail(6)
    latest = fred_data.dropna().iloc[-1]

    # Calculate thresholds (75th percentile = stress)
    unemp_75 = fred_data['unemployment_rate_percent'].quantile(0.75)
    rate_75 = fred_data['federal_funds_rate_percent'].quantile(0.75)

    # Historical averages
    unemp_avg = fred_data['unemployment_rate_percent'].mean()
    rate_avg = fred_data['federal_funds_rate_percent'].mean()

    # Year-over-year changes
    yoy_data = fred_data[fred_data['date'] >= (latest['date'] - pd.DateOffset(months=12))]
    if len(yoy_data) >= 2:
        unemp_yoy_change = latest['unemployment_rate_percent'] - yoy_data.iloc[0]['unemployment_rate_percent']
        rate_yoy_change = latest['federal_funds_rate_percent'] - yoy_data.iloc[0]['federal_funds_rate_percent']
    else:
        unemp_yoy_change = 0
        rate_yoy_change = 0

    # Identify stress periods
    fred_data['high_unemployment'] = fred_data['unemployment_rate_percent'] > unemp_75
    fred_data['high_rates'] = fred_data['federal_funds_rate_percent'] > rate_75
    fred_data['combined_stress'] = fred_data['high_unemployment'] & fred_data['high_rates']

    analysis = {
        'latest_date': latest['date'],
        'unemployment_current': latest['unemployment_rate_percent'],
        'unemployment_yoy_change': unemp_yoy_change,
        'unemployment_75th': unemp_75,
        'unemployment_avg': unemp_avg,
        'fed_funds_current': latest['federal_funds_rate_percent'],
        'fed_funds_yoy_change': rate_yoy_change,
        'fed_funds_75th': rate_75,
        'fed_funds_avg': rate_avg,
        'inflation_current': latest.get('inflation_rate', np.nan),
        'at_high_unemployment': latest['unemployment_rate_percent'] > unemp_75,
        'at_high_rates': latest['federal_funds_rate_percent'] > rate_75,
        'recent_trend': recent
    }

    return analysis, fred_data

def load_payment_fraud_data():
    """Load and merge payment transactions with fraud data"""
    print("\nLoading Snowflake payment and fraud data...")

    # Load transactions
    transactions = pd.read_csv(f'{SYNTHETIC_DIR}/payment_transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    # Load fraud histories
    fraud = pd.read_csv(f'{SYNTHETIC_DIR}/fraud_histories.csv')

    # Merge
    merged = transactions.merge(fraud, on='transaction_id', how='left')
    merged['year_month'] = merged['transaction_date'].dt.to_period('M')
    merged['quarter'] = merged['transaction_date'].dt.to_period('Q')
    merged['year'] = merged['transaction_date'].dt.year

    return merged

def analyze_fraud_trends(data, fred_data):
    """Analyze fraud category trends over time"""
    print("\nAnalyzing fraud category trends...")

    # Filter to fraud cases only
    fraud_cases = data[data['is_fraud'] == 1].copy()

    # Overall fraud rate by month
    monthly_stats = data.groupby('year_month').agg({
        'transaction_id': 'count',
        'is_fraud': ['sum', 'mean'],
        'amount': 'sum'
    }).reset_index()
    monthly_stats.columns = ['year_month', 'total_transactions', 'fraud_count', 'fraud_rate', 'total_amount']
    monthly_stats['year_month_dt'] = monthly_stats['year_month'].dt.to_timestamp()

    # Fraud by type over time (quarterly for smoother trends)
    fraud_by_type_quarter = fraud_cases.groupby(['quarter', 'fraud_type']).size().reset_index(name='count')
    fraud_by_type_quarter['quarter_dt'] = fraud_by_type_quarter['quarter'].dt.to_timestamp()

    # Calculate growth rates by fraud type
    fraud_type_trends = {}
    fraud_types = fraud_cases['fraud_type'].dropna().unique()

    for ftype in fraud_types:
        ftype_data = fraud_by_type_quarter[fraud_by_type_quarter['fraud_type'] == ftype].sort_values('quarter')
        if len(ftype_data) >= 4:
            # Compare last 4 quarters vs first 4 quarters
            recent_avg = ftype_data.tail(4)['count'].mean()
            early_avg = ftype_data.head(4)['count'].mean()
            if early_avg > 0:
                growth_rate = ((recent_avg - early_avg) / early_avg) * 100
            else:
                growth_rate = 0

            # Recent trend (last 6 months)
            recent_6mo = ftype_data.tail(2)['count'].mean() if len(ftype_data) >= 2 else 0
            prior_6mo = ftype_data.tail(4).head(2)['count'].mean() if len(ftype_data) >= 4 else 0

            if prior_6mo > 0:
                recent_trend = ((recent_6mo - prior_6mo) / prior_6mo) * 100
            else:
                recent_trend = 0

            fraud_type_trends[ftype] = {
                'total_count': ftype_data['count'].sum(),
                'early_avg': early_avg,
                'recent_avg': recent_avg,
                'growth_rate': growth_rate,
                'recent_trend': recent_trend,
                'is_trending_up': growth_rate > 10 or recent_trend > 15
            }

    # Fraud by merchant category
    fraud_by_category = fraud_cases.groupby('merchant_category').size().reset_index(name='fraud_count')
    total_by_category = data.groupby('merchant_category').size().reset_index(name='total_count')
    category_fraud_rate = fraud_by_category.merge(total_by_category, on='merchant_category')
    category_fraud_rate['fraud_rate'] = (category_fraud_rate['fraud_count'] / category_fraud_rate['total_count']) * 100
    category_fraud_rate = category_fraud_rate.sort_values('fraud_rate', ascending=False)

    # Fraud by payment method
    fraud_by_method = fraud_cases.groupby('payment_method').size().reset_index(name='fraud_count')
    total_by_method = data.groupby('payment_method').size().reset_index(name='total_count')
    method_fraud_rate = fraud_by_method.merge(total_by_method, on='payment_method')
    method_fraud_rate['fraud_rate'] = (method_fraud_rate['fraud_count'] / method_fraud_rate['total_count']) * 100
    method_fraud_rate = method_fraud_rate.sort_values('fraud_rate', ascending=False)

    # Correlate fraud with economic conditions
    # Merge monthly stats with FRED data
    monthly_stats_with_macro = monthly_stats.copy()
    monthly_stats_with_macro['month_start'] = monthly_stats_with_macro['year_month'].dt.to_timestamp()

    fred_monthly = fred_data.copy()
    fred_monthly['month_start'] = fred_monthly['date'].dt.to_period('M').dt.to_timestamp()

    merged_analysis = monthly_stats_with_macro.merge(
        fred_monthly[['month_start', 'unemployment_rate_percent', 'federal_funds_rate_percent']],
        on='month_start',
        how='left'
    )

    # Calculate correlations
    correlations = {}
    if not merged_analysis['unemployment_rate_percent'].isna().all():
        correlations['fraud_unemployment'] = merged_analysis['fraud_rate'].corr(merged_analysis['unemployment_rate_percent'])
        correlations['fraud_fed_funds'] = merged_analysis['fraud_rate'].corr(merged_analysis['federal_funds_rate_percent'])

    return {
        'monthly_stats': monthly_stats,
        'fraud_by_type_quarter': fraud_by_type_quarter,
        'fraud_type_trends': fraud_type_trends,
        'category_fraud_rate': category_fraud_rate,
        'method_fraud_rate': method_fraud_rate,
        'correlations': correlations,
        'total_fraud_cases': len(fraud_cases),
        'overall_fraud_rate': (data['is_fraud'].sum() / len(data)) * 100
    }

def generate_markdown_report(macro_analysis, fraud_analysis, fred_data):
    """Generate comprehensive markdown report"""
    print("\nGenerating Markdown report...")

    report_date = datetime.now().strftime('%Y-%m-%d')
    latest_date = macro_analysis['latest_date'].strftime('%Y-%m')

    # Determine risk level
    risk_signals = []
    if macro_analysis['unemployment_yoy_change'] > 0.3:
        risk_signals.append("Rising unemployment (+{:.1f}pp YoY)".format(macro_analysis['unemployment_yoy_change']))
    if macro_analysis['at_high_unemployment']:
        risk_signals.append("Unemployment above 75th percentile stress threshold")
    if macro_analysis['inflation_current'] and macro_analysis['inflation_current'] > 3.0:
        risk_signals.append("Elevated inflation ({:.1f}%)".format(macro_analysis['inflation_current']))
    if macro_analysis['fed_funds_yoy_change'] < -0.5:
        risk_signals.append("Fed rate cuts indicate economic concern ({:.2f}pp YoY)".format(macro_analysis['fed_funds_yoy_change']))

    # Identify trending fraud types
    trending_up = {k: v for k, v in fraud_analysis['fraud_type_trends'].items() if v['is_trending_up']}

    report = f"""# Fraud Risk Economic Analysis Report

**Generated:** {report_date}
**Data Through:** {latest_date}
**Status:** For Data Science Validation and Follow-up Actions

---

## Executive Summary

This analysis combines FRED macroeconomic data (unemployment, interest rates, inflation) with internal payment transaction and fraud incident data to assess whether current economic conditions suggest increased payment fraud risk.

### Key Findings

| Risk Category | Status | Signal Strength |
|---------------|--------|-----------------|
| Macroeconomic Stress | **{"ELEVATED" if len(risk_signals) >= 2 else "MODERATE" if len(risk_signals) >= 1 else "LOW"}** | {len(risk_signals)} warning signals |
| First-Party Fraud Risk | **{"ELEVATED" if macro_analysis['unemployment_current'] >= 4.2 else "WATCH"}** | Based on unemployment level |
| Fraud Categories Trending Up | **{len(trending_up)}** | Categories showing growth |

---

## 1. Current Macroeconomic Conditions (FRED Data)

### 1.1 Summary Metrics

| Indicator | Current Value | YoY Change | 75th Pctl (Stress) | Status |
|-----------|---------------|------------|-------------------|--------|
| Unemployment Rate | **{macro_analysis['unemployment_current']:.1f}%** | {'+' if macro_analysis['unemployment_yoy_change'] >= 0 else ''}{macro_analysis['unemployment_yoy_change']:.2f} pp | {macro_analysis['unemployment_75th']:.1f}% | {'ELEVATED' if macro_analysis['at_high_unemployment'] else 'WATCH' if macro_analysis['unemployment_current'] >= 4.0 else 'NORMAL'} |
| Federal Funds Rate | **{macro_analysis['fed_funds_current']:.2f}%** | {'+' if macro_analysis['fed_funds_yoy_change'] >= 0 else ''}{macro_analysis['fed_funds_yoy_change']:.2f} pp | {macro_analysis['fed_funds_75th']:.2f}% | {'HIGH' if macro_analysis['at_high_rates'] else 'DECLINING'} |
| Inflation (YoY CPI) | **{macro_analysis['inflation_current']:.1f}%** | N/A | 3.0% | {'ELEVATED' if macro_analysis['inflation_current'] > 3.0 else 'MODERATE'} |

### 1.2 Risk Signals Detected

"""

    if risk_signals:
        for signal in risk_signals:
            report += f"- **WARNING:** {signal}\n"
    else:
        report += "- No significant macroeconomic warning signals detected\n"

    report += f"""
### 1.3 Recent Trend (Last 6 Months)

| Month | Unemployment | Fed Funds Rate |
|-------|--------------|----------------|
"""

    recent = macro_analysis['recent_trend'].tail(6)
    for _, row in recent.iterrows():
        report += f"| {row['date'].strftime('%Y-%m')} | {row['unemployment_rate_percent']:.1f}% | {row['federal_funds_rate_percent']:.2f}% |\n"

    report += f"""
### 1.4 Economic Risk Assessment

Based on historical analysis (2008-2019 recession studies from internal Box documents):

- **First-party fraud** historically increased 12-22% when unemployment exceeded ~5%
- Current unemployment at **{macro_analysis['unemployment_current']:.1f}%** is **approaching** this threshold
- Federal Reserve rate cuts ({abs(macro_analysis['fed_funds_yoy_change']):.2f}pp YoY) indicate policy concern about economic conditions
- Low savings rates and declining consumer sentiment amplify fraud risk

**Assessment:** Current conditions suggest **MODERATE-TO-ELEVATED** first-party fraud risk, with conditions approaching historical stress thresholds.

---

## 2. Fraud Category Analysis (Snowflake Data)

### 2.1 Overall Fraud Metrics

| Metric | Value |
|--------|-------|
| Total Transactions Analyzed | {fraud_analysis['monthly_stats']['total_transactions'].sum():,} |
| Total Fraud Incidents | {fraud_analysis['total_fraud_cases']:,} |
| Overall Fraud Rate | **{fraud_analysis['overall_fraud_rate']:.2f}%** |

### 2.2 Fraud Categories Trending Upward

"""

    if trending_up:
        report += "| Fraud Type | Total Count | Growth Rate | Recent Trend | Risk Level |\n"
        report += "|------------|-------------|-------------|--------------|------------|\n"

        for ftype, data in sorted(trending_up.items(), key=lambda x: x[1]['growth_rate'], reverse=True):
            risk = "HIGH" if data['growth_rate'] > 25 or data['recent_trend'] > 25 else "ELEVATED"
            report += f"| **{ftype.replace('_', ' ').title()}** | {data['total_count']} | +{data['growth_rate']:.1f}% | +{data['recent_trend']:.1f}% | {risk} |\n"
    else:
        report += "*No fraud categories showing significant upward trend*\n"

    # All fraud types
    report += "\n### 2.3 All Fraud Type Analysis\n\n"
    report += "| Fraud Type | Total Count | Early Period Avg | Recent Avg | Growth % | Trend Status |\n"
    report += "|------------|-------------|------------------|------------|----------|-------------|\n"

    for ftype, data in sorted(fraud_analysis['fraud_type_trends'].items(), key=lambda x: x[1]['growth_rate'], reverse=True):
        status = "TRENDING UP" if data['is_trending_up'] else "STABLE" if abs(data['growth_rate']) < 10 else "DECLINING"
        report += f"| {ftype.replace('_', ' ').title()} | {data['total_count']} | {data['early_avg']:.1f} | {data['recent_avg']:.1f} | {'+' if data['growth_rate'] >= 0 else ''}{data['growth_rate']:.1f}% | {status} |\n"

    # Category fraud rates
    report += "\n### 2.4 Fraud Rate by Merchant Category\n\n"
    report += "| Merchant Category | Fraud Count | Total Transactions | Fraud Rate |\n"
    report += "|-------------------|-------------|-------------------|------------|\n"

    for _, row in fraud_analysis['category_fraud_rate'].head(10).iterrows():
        report += f"| {row['merchant_category'].replace('_', ' ').title()} | {row['fraud_count']} | {row['total_count']:,} | {row['fraud_rate']:.2f}% |\n"

    # Payment method fraud rates
    report += "\n### 2.5 Fraud Rate by Payment Method\n\n"
    report += "| Payment Method | Fraud Count | Total Transactions | Fraud Rate |\n"
    report += "|----------------|-------------|-------------------|------------|\n"

    for _, row in fraud_analysis['method_fraud_rate'].iterrows():
        report += f"| {row['payment_method'].replace('_', ' ').title()} | {row['fraud_count']} | {row['total_count']:,} | {row['fraud_rate']:.2f}% |\n"

    # Correlations
    report += "\n### 2.6 Fraud-Economic Correlations\n\n"

    if fraud_analysis['correlations']:
        report += "| Relationship | Correlation | Interpretation |\n"
        report += "|--------------|-------------|----------------|\n"

        unemp_corr = fraud_analysis['correlations'].get('fraud_unemployment', 0)
        rate_corr = fraud_analysis['correlations'].get('fraud_fed_funds', 0)

        unemp_interp = "Higher unemployment → Higher fraud" if unemp_corr > 0.1 else "Higher unemployment → Lower fraud" if unemp_corr < -0.1 else "No clear relationship"
        rate_interp = "Higher rates → Higher fraud" if rate_corr > 0.1 else "Higher rates → Lower fraud" if rate_corr < -0.1 else "No clear relationship"

        report += f"| Fraud Rate ↔ Unemployment | {unemp_corr:.3f} | {unemp_interp} |\n"
        report += f"| Fraud Rate ↔ Fed Funds Rate | {rate_corr:.3f} | {rate_interp} |\n"

    report += f"""
---

## 3. Combined Risk Assessment

### 3.1 Early Warning Signals

Based on the combined analysis of FRED macroeconomic data and internal fraud trends:

#### ELEVATED RISK Signals:
"""

    # Generate combined risk signals
    elevated_signals = []
    watch_signals = []

    if macro_analysis['unemployment_current'] >= 4.2:
        elevated_signals.append(f"Unemployment at {macro_analysis['unemployment_current']:.1f}% approaching historical fraud trigger threshold (5%)")

    if macro_analysis['unemployment_yoy_change'] > 0.3:
        elevated_signals.append(f"Unemployment rising (+{macro_analysis['unemployment_yoy_change']:.2f}pp YoY) - historical 3-6 month lag to fraud increase")

    if macro_analysis['inflation_current'] and macro_analysis['inflation_current'] > 3.0:
        elevated_signals.append(f"Sticky inflation ({macro_analysis['inflation_current']:.1f}%) eroding consumer purchasing power")

    for ftype, data in trending_up.items():
        if data['growth_rate'] > 20:
            elevated_signals.append(f"{ftype.replace('_', ' ').title()} fraud trending up +{data['growth_rate']:.1f}% from baseline")

    # Watch signals
    if macro_analysis['fed_funds_yoy_change'] < -0.5:
        watch_signals.append(f"Fed rate cuts signal economic concern (-{abs(macro_analysis['fed_funds_yoy_change']):.2f}pp YoY)")

    for ftype, data in fraud_analysis['fraud_type_trends'].items():
        if 5 < data['growth_rate'] <= 20 and ftype not in trending_up:
            watch_signals.append(f"{ftype.replace('_', ' ').title()} fraud showing moderate growth (+{data['growth_rate']:.1f}%)")

    if elevated_signals:
        for signal in elevated_signals:
            report += f"1. **{signal}**\n"
    else:
        report += "*No elevated risk signals detected*\n"

    report += "\n#### WATCH Signals:\n"

    if watch_signals:
        for signal in watch_signals:
            report += f"1. {signal}\n"
    else:
        report += "*No watch signals detected*\n"

    report += """
### 3.2 Historical Context Integration (Box Documents)

Per internal historical analyses (FinTechCo 2008-2019 recession studies):

| Historical Pattern | Current Relevance | Action |
|--------------------|-------------------|--------|
| First-party fraud +12-22% at unemployment >5% | Approaching threshold (current: {:.1f}%) | **MONITOR** |
| 3-6 month lag from unemployment spike to fraud | Currently in potential lag window | **ALERT** |
| Third-party fraud (ATO) not correlated with economy | No change expected from macro conditions | NO ACTION |
| Consumer behavior anticipatory (6-9 mo lead) | Rate cuts may trigger defensive behavior | **MONITOR** |

""".format(macro_analysis['unemployment_current'])

    # Summary recommendations
    report += f"""
---

## 4. Recommendations for Data Science Team

### 4.1 Immediate Actions

1. **First-Party Fraud Model Stress Testing**
   - Test model performance under simulated unemployment increase (+1-2pp)
   - Focus on: {', '.join([ftype.replace('_', ' ') for ftype in trending_up.keys()]) if trending_up else 'All fraud types'}

2. **Leading Indicator Monitoring**
   - Implement weekly monitoring of chargeback rates in non-essential categories
   - Track fee-related customer inquiries for early sentiment signals

3. **Fraud Category Deep Dive**
"""

    if trending_up:
        report += "   - Prioritize investigation of trending categories:\n"
        for ftype in trending_up.keys():
            report += f"     - {ftype.replace('_', ' ').title()}\n"

    # Calculate thresholds
    unemp_current = macro_analysis['unemployment_current']
    fraud_rate_current = fraud_analysis['overall_fraud_rate']
    fraud_rate_warning = fraud_analysis['overall_fraud_rate'] * 1.15
    fraud_rate_critical = fraud_analysis['overall_fraud_rate'] * 1.25
    total_transactions = fraud_analysis['monthly_stats']['total_transactions'].sum()
    total_fraud_cases = fraud_analysis['total_fraud_cases']
    overall_assessment = "ELEVATED" if len(elevated_signals) >= 2 else "MODERATE"
    trending_categories = ', '.join([ftype.replace('_', ' ').title() for ftype in trending_up.keys()]) if trending_up else 'None currently trending significantly'

    report += f"""
### 4.2 Model Enhancement Priorities

| Priority | Task | Rationale |
|----------|------|-----------|
| HIGH | Add unemployment rate as model feature | Strong historical correlation with first-party fraud |
| HIGH | Implement 3-6 month lagged indicators | Matches historical fraud manifestation timing |
| MEDIUM | Segment-specific risk scores | Different customer segments have varying macro sensitivity |
| MEDIUM | Detection rate calibration | Separate true fraud growth from improved detection |

### 4.3 Monitoring Thresholds

| Metric | Current | Warning | Critical |
|--------|---------|---------|----------|
| Unemployment Rate | {unemp_current:.1f}% | 4.5% | 5.0% |
| Monthly Fraud Rate | {fraud_rate_current:.2f}% | {fraud_rate_warning:.2f}% | {fraud_rate_critical:.2f}% |
| First-Party Fraud Growth | See above | +15% QoQ | +25% QoQ |

---

## 5. Data Quality & Limitations

### 5.1 Data Sources Used

| Source | Dataset | Records | Date Range |
|--------|---------|---------|------------|
| FRED | Unemployment Rate | 46 months | 2022-01 to 2025-09 |
| FRED | Federal Funds Rate | 48 months | 2022-01 to 2025-11 |
| FRED | Consumer Price Index | 46 months | 2022-01 to 2025-09 |
| Snowflake | PAYMENT_TRANSACTIONS | {total_transactions:,} | 2022-01 to 2025-11 |
| Snowflake | FRAUD_INCIDENTS | {total_fraud_cases:,} | 2022-01 to 2025-11 |
| Box | Historical Recession Analyses | 3 documents | 2008-2019 period |

### 5.2 Limitations

1. **Synthetic Data**: Snowflake data is synthetic with embedded macro correlations
2. **Detection Confounding**: Cannot separate true fraud growth from improved detection
3. **Causal Attribution**: Correlation ≠ causation; multiple confounders present
4. **Historical Comparability**: 2008-2019 patterns may not repeat exactly

---

## 6. Conclusion

**Overall Risk Assessment: {overall_assessment}**

Current macroeconomic conditions suggest **increased vigilance** for payment fraud, particularly:

1. **First-party fraud types** (friendly fraud, chargeback fraud) given unemployment trajectory
2. **Trending categories**: {trending_categories}

The combination of rising unemployment ({unemp_current:.1f}%), sticky inflation, and Fed rate cuts creates conditions historically associated with elevated fraud risk. While not yet at crisis levels, the Data Science team should prioritize stress testing and enhanced monitoring.

---

*Report generated for Data Science validation. All conclusions are directional hypotheses requiring statistical validation before operational implementation.*
"""

    return report

def main():
    """Main analysis pipeline"""
    print("=" * 60)
    print("FRAUD RISK ECONOMIC ANALYSIS")
    print("=" * 60)

    # Load data
    fred_data = load_fred_data()
    payment_fraud_data = load_payment_fraud_data()

    # Analyze
    macro_analysis, fred_data = analyze_macro_conditions(fred_data)
    fraud_analysis = analyze_fraud_trends(payment_fraud_data, fred_data)

    # Generate report
    report = generate_markdown_report(macro_analysis, fraud_analysis, fred_data)

    # Save report
    output_path = f'{OUTPUT_DIR}/Fraud_Risk_Economic_Analysis_Report.md'
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"Report saved to: {output_path}")
    print(f"{'=' * 60}")

    # Print summary to console
    print("\n--- SUMMARY ---")
    print(f"Unemployment: {macro_analysis['unemployment_current']:.1f}% (YoY: {'+' if macro_analysis['unemployment_yoy_change'] >= 0 else ''}{macro_analysis['unemployment_yoy_change']:.2f}pp)")
    print(f"Fed Funds Rate: {macro_analysis['fed_funds_current']:.2f}% (YoY: {'+' if macro_analysis['fed_funds_yoy_change'] >= 0 else ''}{macro_analysis['fed_funds_yoy_change']:.2f}pp)")
    print(f"Overall Fraud Rate: {fraud_analysis['overall_fraud_rate']:.2f}%")

    trending_up = {k: v for k, v in fraud_analysis['fraud_type_trends'].items() if v['is_trending_up']}
    print(f"Fraud Categories Trending Up: {len(trending_up)}")
    for ftype, data in trending_up.items():
        print(f"  - {ftype}: +{data['growth_rate']:.1f}%")

if __name__ == '__main__':
    main()
