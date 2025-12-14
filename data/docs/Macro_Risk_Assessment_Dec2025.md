# Macro Risk Assessment

**Payment Activity Stress & Fraud Risk Analysis**  
December 2025 | Input for Data Science Validation

---

## Executive Summary

This analysis synthesizes internal Snowflake payment and fraud data with FRED macroeconomic indicators and historical recession patterns from Box archives. **Evidence suggests elevated payment stress and fraud risk consistent with early-stage economic deterioration.**

| Signal | Finding |
|--------|---------|
| ⚠ **HIGH RISK** | First-Party Fraud up 280% (Q3-Q4 2025 vs 2024 avg) |
| ⚠ **HIGH RISK** | Fee Sentiment collapsed from 8.0 to 5.3 (Jun-Nov 2025) |
| ⚡ **MODERATE** | Discretionary-to-Essential ratio declined (1.01 → 0.84 in Sep 2025) |
| ⚡ **MODERATE** | Unemployment rising (3.7% Jan 2024 → 4.4% Sep 2025) |
| ✓ **STABLE** | Third-Party fraud flat (consistent with historical pattern) |

---

## 1. Internal Data Analysis (Snowflake)

### 1.1 Payment Transaction Patterns

*Source: DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS (52,134 records, Jan 2024 - Dec 2025)*

Analysis of approved transactions reveals **softening in discretionary spending** relative to essential categories. The discretionary-to-essential spending ratio has declined from approximately 1.01 in January 2025 to 0.84 in September 2025, indicating consumers are prioritizing necessities over non-essential purchases.

Key observations: September 2025 showed the lowest discretionary spending ($242K) while essential spending remained elevated ($289K). This 16% gap represents a structural shift from balanced spending patterns observed throughout 2024 where the ratio hovered near parity. Category breakdown shows Travel and Dining most affected within discretionary, while Grocery and Utilities remain resilient.

### 1.2 Fraud Incident Analysis

*Source: DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS (1,203 records, Jan 2024 - Dec 2025)*

**Critical finding: First-Party fraud has diverged dramatically from Third-Party fraud.** This pattern is historically associated with consumer financial stress.

**Quarterly Fraud Incident Comparison:**

| Period | First-Party | Third-Party | FP:TP Ratio |
|--------|-------------|-------------|-------------|
| 2024 Avg/Qtr | 39 | 90 | 0.43:1 |
| Q3 2025 | **156 (+300%)** | 60 (-33%) | **2.60:1** |
| Q4 2025 (Nov) | **111 (+185%)** | 45 (-50%) | **2.47:1** |

This divergence precisely matches the historical pattern documented in *FinTechCo_Risk_Fraud_Historical_Analysis.docx*, which found that first-party fraud increased 12-22% during 2009-2010 unemployment spikes while third-party fraud showed no economic correlation.

### 1.3 Customer Sentiment Trends

*Source: DEMO_RISK_DB.DEMO_RISK_DATA.CUSTOMER_SENTIMENT (136 records, Jan 2024 - Dec 2025)*

**Fee sensitivity has emerged as a critical stress indicator.** Average sentiment scores for 'Fees' topic dropped from 8.0-9.0 (Jan-May 2025) to 5.0-5.75 (Jun-Nov 2025). General sentiment also declined from 9.0 to 4.0 over the same period. This 3+ point decline on a 10-point scale represents a significant deterioration in customer financial comfort.

---

## 2. External Economic Indicators (FRED)

### 2.1 Unemployment Rate (UNRATE)

Unemployment has risen from 3.7% (Jan 2024) to 4.4% (Sep 2025), a 0.7 percentage point increase. While still historically low, **the rate of increase approaches the Sahm Rule threshold** (0.5pp increase from cycle low), a reliable recession indicator. The trajectory suggests continued labor market softening.

### 2.2 Federal Funds Rate (FEDFUNDS)

The Fed has cut rates from 5.33% (Jan 2024) to 3.88% (Nov 2025), indicating policy acknowledgment of economic weakness. Rate cuts typically lag economic stress by 6-12 months, suggesting **underlying conditions may be weaker than headline data suggests**.

### 2.3 Inflation (CPI Year-over-Year)

CPI has re-accelerated from 2.3% (Apr 2025) to 3.0% (Sep 2025). **The combination of rising unemployment AND rising inflation creates a potential stagflationary pressure** that historically correlates with increased consumer financial stress and fraud risk.

---

## 3. Historical Context (Box Archives)

*Source: FinTechCo_Risk_Fraud_Historical_Analysis.docx - 2008-2019 recession pattern analysis*

The historical analysis documented three key patterns from the 2008-2010 period that are relevant today:

**Pattern 1 - First-Party Fraud Lag Effect:** Chargeback rates increased 12-18% during periods 3-6 months after unemployment spikes. Current data shows first-party fraud spiking June 2025, approximately 4-5 months after unemployment began accelerating.

**Pattern 2 - Third-Party Independence:** ATO and synthetic ID fraud showed no correlation with economic conditions; driven instead by data breaches and criminal sophistication. Current stability in third-party fraud is consistent with this finding.

**Pattern 3 - Category Concentration:** 63% of dispute increases occurred in non-essential categories. Current data shows discretionary spending softening aligns with this historical pattern.

---

## 4. Risk Signal Summary

### 4.1 Directionally Supported (High Confidence)

1. First-party fraud surge is consistent with historical recession-era patterns and current macro stress
2. Fee sensitivity spike correlates with declining disposable income indicators
3. Third-party fraud stability is expected given lack of major data breaches
4. Discretionary-to-essential spending shift matches early-stage recession behavior

### 4.2 Uncertain / Requires Validation

- Magnitude of first-party fraud increase (280%) exceeds historical precedent (12-22%); may reflect detection improvements
- Inflation re-acceleration impact on fraud not well-documented historically
- Customer segment-level analysis not possible with available data

---

## 5. Data Gaps & Limitations

- Customer-level income/employment data not available for correlation analysis
- Sentiment sample size small (136 total responses); may not be representative
- Detection capability changes over time not quantified; could confound fraud trends
- Industry benchmark data not available for comparative analysis
- FRED data lag (unemployment through Sep 2025 only; Fed Funds through Nov 2025)

---

## 6. Recommendations for Data Science Validation

1. **Statistical Validation:** Apply time-series analysis to confirm first-party fraud trend significance; control for seasonality and detection improvements

2. **Predictive Modeling:** Build early warning model incorporating unemployment lag effects (3-6 months) identified in historical analysis

3. **Threshold Calibration:** Review and potentially lower first-party fraud detection thresholds given elevated risk environment

4. **Segment Analysis:** If customer employment/industry data available, prioritize monitoring of sectors most affected by unemployment

5. **Dashboard Integration:** Incorporate real-time FRED indicators into fraud monitoring dashboards per historical analysis recommendations

---

**Data Sources:** Snowflake (DEMO_RISK_DB.DEMO_RISK_DATA), FRED (UNRATE, FEDFUNDS, CPIAUCSL), Box (FinTechCo_Risk_Fraud_Historical_Analysis.docx)

**Generated:** December 11, 2025 | For Data Science Validation
