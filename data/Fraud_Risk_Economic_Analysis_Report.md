# Fraud Risk Economic Analysis Report

**Generated:** 2025-12-14
**Data Through:** 2025-09
**Status:** For Data Science Validation and Follow-up Actions

---

## Executive Summary

This analysis combines FRED macroeconomic data (unemployment, interest rates, inflation) with internal payment transaction and fraud incident data to assess whether current economic conditions suggest increased payment fraud risk.

### Key Findings

| Risk Category | Status | Signal Strength |
|---------------|--------|-----------------|
| Macroeconomic Stress | **ELEVATED** | 4 warning signals |
| First-Party Fraud Risk | **ELEVATED** | Based on unemployment level |
| Fraud Categories Trending Up | **1** | Categories showing growth |

---

## 1. Current Macroeconomic Conditions (FRED Data)

### 1.1 Summary Metrics

| Indicator | Current Value | YoY Change | 75th Pctl (Stress) | Status |
|-----------|---------------|------------|-------------------|--------|
| Unemployment Rate | **4.4%** | +0.30 pp | 4.1% | ELEVATED |
| Federal Funds Rate | **4.22%** | -0.91 pp | 5.33% | DECLINING |
| Inflation (YoY CPI) | **3.0%** | N/A | 3.0% | ELEVATED |

### 1.2 Risk Signals Detected

- **WARNING:** Rising unemployment (+0.3pp YoY)
- **WARNING:** Unemployment above 75th percentile stress threshold
- **WARNING:** Elevated inflation (3.0%)
- **WARNING:** Fed rate cuts indicate economic concern (-0.91pp YoY)

### 1.3 Recent Trend (Last 6 Months)

| Month | Unemployment | Fed Funds Rate |
|-------|--------------|----------------|
| 2025-04 | 4.2% | 4.33% |
| 2025-05 | 4.2% | 4.33% |
| 2025-06 | 4.1% | 4.33% |
| 2025-07 | 4.2% | 4.33% |
| 2025-08 | 4.3% | 4.33% |
| 2025-09 | 4.4% | 4.22% |

### 1.4 Economic Risk Assessment

Based on historical analysis (2008-2019 recession studies from internal Box documents):

- **First-party fraud** historically increased 12-22% when unemployment exceeded ~5%
- Current unemployment at **4.4%** is **approaching** this threshold
- Federal Reserve rate cuts (0.91pp YoY) indicate policy concern about economic conditions
- Low savings rates and declining consumer sentiment amplify fraud risk

**Assessment:** Current conditions suggest **MODERATE-TO-ELEVATED** first-party fraud risk, with conditions approaching historical stress thresholds.

---

## 2. Fraud Category Analysis (Snowflake Data)

### 2.1 Overall Fraud Metrics

| Metric | Value |
|--------|-------|
| Total Transactions Analyzed | 50,000 |
| Total Fraud Incidents | 909 |
| Overall Fraud Rate | **1.82%** |

### 2.2 Fraud Categories Trending Upward

| Fraud Type | Total Count | Growth Rate | Recent Trend | Risk Level |
|------------|-------------|-------------|--------------|------------|
| **Chargeback Fraud** | 179 | +22.7% | +16.0% | ELEVATED |

### 2.3 All Fraud Type Analysis

| Fraud Type | Total Count | Early Period Avg | Recent Avg | Growth % | Trend Status |
|------------|-------------|------------------|------------|----------|-------------|
| Chargeback Fraud | 179 | 11.0 | 13.5 | +22.7% | TRENDING UP |
| Synthetic Identity | 186 | 12.5 | 12.8 | +2.0% | STABLE |
| Friendly Fraud | 184 | 13.0 | 12.0 | -7.7% | STABLE |
| Account Takeover | 189 | 14.8 | 12.2 | -16.9% | DECLINING |
| Stolen Card | 171 | 13.0 | 8.0 | -38.5% | DECLINING |

### 2.4 Fraud Rate by Merchant Category

| Merchant Category | Fraud Count | Total Transactions | Fraud Rate |
|-------------------|-------------|-------------------|------------|
| Travel | 54 | 2,500 | 2.16% |
| Other | 55 | 2,547 | 2.16% |
| Healthcare | 53 | 2,559 | 2.07% |
| Retail | 50 | 2,532 | 1.97% |
| Online Services | 49 | 2,511 | 1.95% |
| Education | 49 | 2,522 | 1.94% |
| P2P Transfer | 186 | 9,985 | 1.86% |
| Bill Payment | 32 | 1,743 | 1.84% |
| Loan Payment | 31 | 1,709 | 1.81% |
| Entertainment | 44 | 2,459 | 1.79% |

### 2.5 Fraud Rate by Payment Method

| Payment Method | Fraud Count | Total Transactions | Fraud Rate |
|----------------|-------------|-------------------|------------|
| Debit Card | 297 | 15,024 | 1.98% |
| Cash | 46 | 2,473 | 1.86% |
| Digital Wallet | 89 | 4,933 | 1.80% |
| Credit Card | 352 | 20,045 | 1.76% |
| Bank Transfer | 125 | 7,525 | 1.66% |

### 2.6 Fraud-Economic Correlations

| Relationship | Correlation | Interpretation |
|--------------|-------------|----------------|
| Fraud Rate ↔ Unemployment | 0.508 | Higher unemployment → Higher fraud |
| Fraud Rate ↔ Fed Funds Rate | -0.031 | No clear relationship |

---

## 3. Combined Risk Assessment

### 3.1 Early Warning Signals

Based on the combined analysis of FRED macroeconomic data and internal fraud trends:

#### ELEVATED RISK Signals:
1. **Unemployment at 4.4% approaching historical fraud trigger threshold (5%)**
1. **Unemployment rising (+0.30pp YoY) - historical 3-6 month lag to fraud increase**
1. **Sticky inflation (3.0%) eroding consumer purchasing power**
1. **Chargeback Fraud fraud trending up +22.7% from baseline**

#### WATCH Signals:
1. Fed rate cuts signal economic concern (-0.91pp YoY)

### 3.2 Historical Context Integration (Box Documents)

Per internal historical analyses (FinTechCo 2008-2019 recession studies):

| Historical Pattern | Current Relevance | Action |
|--------------------|-------------------|--------|
| First-party fraud +12-22% at unemployment >5% | Approaching threshold (current: 4.4%) | **MONITOR** |
| 3-6 month lag from unemployment spike to fraud | Currently in potential lag window | **ALERT** |
| Third-party fraud (ATO) not correlated with economy | No change expected from macro conditions | NO ACTION |
| Consumer behavior anticipatory (6-9 mo lead) | Rate cuts may trigger defensive behavior | **MONITOR** |


---

## 4. Recommendations for Data Science Team

### 4.1 Immediate Actions

1. **First-Party Fraud Model Stress Testing**
   - Test model performance under simulated unemployment increase (+1-2pp)
   - Focus on: chargeback fraud

2. **Leading Indicator Monitoring**
   - Implement weekly monitoring of chargeback rates in non-essential categories
   - Track fee-related customer inquiries for early sentiment signals

3. **Fraud Category Deep Dive**
   - Prioritize investigation of trending categories:
     - Chargeback Fraud

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
| Unemployment Rate | 4.4% | 4.5% | 5.0% |
| Monthly Fraud Rate | 1.82% | 2.09% | 2.27% |
| First-Party Fraud Growth | See above | +15% QoQ | +25% QoQ |

---

## 5. Data Quality & Limitations

### 5.1 Data Sources Used

| Source | Dataset | Records | Date Range |
|--------|---------|---------|------------|
| FRED | Unemployment Rate | 46 months | 2022-01 to 2025-09 |
| FRED | Federal Funds Rate | 48 months | 2022-01 to 2025-11 |
| FRED | Consumer Price Index | 46 months | 2022-01 to 2025-09 |
| Snowflake | PAYMENT_TRANSACTIONS | 50,000 | 2022-01 to 2025-11 |
| Snowflake | FRAUD_INCIDENTS | 909 | 2022-01 to 2025-11 |
| Box | Historical Recession Analyses | 3 documents | 2008-2019 period |

### 5.2 Limitations

1. **Synthetic Data**: Snowflake data is synthetic with embedded macro correlations
2. **Detection Confounding**: Cannot separate true fraud growth from improved detection
3. **Causal Attribution**: Correlation ≠ causation; multiple confounders present
4. **Historical Comparability**: 2008-2019 patterns may not repeat exactly

---

## 6. Conclusion

**Overall Risk Assessment: ELEVATED**

Current macroeconomic conditions suggest **increased vigilance** for payment fraud, particularly:

1. **First-party fraud types** (friendly fraud, chargeback fraud) given unemployment trajectory
2. **Trending categories**: Chargeback Fraud

The combination of rising unemployment (4.4%), sticky inflation, and Fed rate cuts creates conditions historically associated with elevated fraud risk. While not yet at crisis levels, the Data Science team should prioritize stress testing and enhanced monitoring.

---

*Report generated for Data Science validation. All conclusions are directional hypotheses requiring statistical validation before operational implementation.*
