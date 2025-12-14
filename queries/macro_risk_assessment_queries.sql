-- ============================================================================
-- MACRO RISK ASSESSMENT - SNOWFLAKE QUERIES
-- Purpose: Payment fraud & economic stress analysis
-- Date: December 2025
-- Database: DEMO_RISK_DB.DEMO_RISK_DATA
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. PAYMENT TRANSACTIONS ANALYSIS
-- ----------------------------------------------------------------------------

-- 1.1 Basic record count and date range
SELECT 
    COUNT(*) AS total_records, 
    MIN(TXN_DATE) AS earliest_date, 
    MAX(TXN_DATE) AS latest_date 
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS;

-- 1.2 Monthly transaction counts
SELECT 
    YEAR(TXN_DATE) AS yr, 
    MONTH(TXN_DATE) AS mo, 
    COUNT(*) AS txn_count
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS
GROUP BY 1, 2
ORDER BY 1, 2;

-- 1.3 Distinct categories
SELECT DISTINCT CATEGORY 
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS;

-- 1.4 Total spending by category (all time)
SELECT 
    CATEGORY, 
    ROUND(SUM(AMOUNT), 0) AS total_amount
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS
WHERE STATUS = 'APPROVED'
GROUP BY CATEGORY
ORDER BY CATEGORY;

-- 1.5 Quarterly spending by category (YoY comparison)
SELECT 
    CATEGORY, 
    ROUND(SUM(CASE WHEN TXN_DATE BETWEEN '2024-01-01' AND '2024-12-31' THEN AMOUNT ELSE 0 END), 0) AS Y2024,
    ROUND(SUM(CASE WHEN TXN_DATE BETWEEN '2025-01-01' AND '2025-03-31' THEN AMOUNT ELSE 0 END), 0) AS Q1_2025,
    ROUND(SUM(CASE WHEN TXN_DATE BETWEEN '2025-04-01' AND '2025-06-30' THEN AMOUNT ELSE 0 END), 0) AS Q2_2025,
    ROUND(SUM(CASE WHEN TXN_DATE BETWEEN '2025-07-01' AND '2025-09-30' THEN AMOUNT ELSE 0 END), 0) AS Q3_2025,
    ROUND(SUM(CASE WHEN TXN_DATE >= '2025-10-01' THEN AMOUNT ELSE 0 END), 0) AS Q4_2025
FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS
WHERE STATUS = 'APPROVED'
GROUP BY CATEGORY
ORDER BY CATEGORY;


-- ----------------------------------------------------------------------------
-- 2. FRAUD INCIDENTS ANALYSIS
-- ----------------------------------------------------------------------------

-- 2.1 Distinct fraud types
SELECT DISTINCT TYPE 
FROM DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS;

-- 2.2 Quarterly fraud incident counts by type
SELECT 
    TYPE, 
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2024-01-01' AND '2024-03-31' THEN 1 END) AS Q1_2024,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2024-04-01' AND '2024-06-30' THEN 1 END) AS Q2_2024,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2024-07-01' AND '2024-09-30' THEN 1 END) AS Q3_2024,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2024-10-01' AND '2024-12-31' THEN 1 END) AS Q4_2024,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2025-01-01' AND '2025-03-31' THEN 1 END) AS Q1_2025,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2025-04-01' AND '2025-06-30' THEN 1 END) AS Q2_2025,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2025-07-01' AND '2025-09-30' THEN 1 END) AS Q3_2025,
    COUNT(CASE WHEN INCIDENT_DATE >= '2025-10-01' THEN 1 END) AS Q4_2025
FROM DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS
GROUP BY TYPE
ORDER BY TYPE;

-- 2.3 Fraud incidents and amounts by type (YoY comparison)
SELECT 
    TYPE, 
    ROUND(SUM(CASE WHEN INCIDENT_DATE BETWEEN '2024-01-01' AND '2024-12-31' THEN AMOUNT ELSE 0 END), 0) AS AMT_2024,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2024-01-01' AND '2024-12-31' THEN 1 END) AS CNT_2024,
    ROUND(SUM(CASE WHEN INCIDENT_DATE BETWEEN '2025-01-01' AND '2025-12-31' THEN AMOUNT ELSE 0 END), 0) AS AMT_2025,
    COUNT(CASE WHEN INCIDENT_DATE BETWEEN '2025-01-01' AND '2025-12-31' THEN 1 END) AS CNT_2025
FROM DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS
GROUP BY TYPE
ORDER BY TYPE;

-- 2.4 Monthly fraud detail for 2025 (trend analysis)
SELECT 
    YEAR(INCIDENT_DATE) AS yr, 
    MONTH(INCIDENT_DATE) AS mo, 
    TYPE, 
    COUNT(*) AS incident_count, 
    ROUND(SUM(AMOUNT), 0) AS total_amount
FROM DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS
WHERE INCIDENT_DATE >= '2025-01-01'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;


-- ----------------------------------------------------------------------------
-- 3. CUSTOMER SENTIMENT ANALYSIS
-- ----------------------------------------------------------------------------

-- 3.1 Monthly sentiment scores by topic (2025)
SELECT 
    YEAR(SURVEY_DATE) AS yr, 
    MONTH(SURVEY_DATE) AS mo, 
    TOPIC, 
    ROUND(AVG(SENTIMENT_SCORE), 2) AS avg_score, 
    COUNT(*) AS response_count
FROM DEMO_RISK_DB.DEMO_RISK_DATA.CUSTOMER_SENTIMENT
WHERE SURVEY_DATE >= '2025-01-01'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;


-- ----------------------------------------------------------------------------
-- 4. DERIVED METRICS FOR RISK ASSESSMENT
-- ----------------------------------------------------------------------------

-- 4.1 Discretionary vs Essential spending ratio by quarter
WITH categorized AS (
    SELECT 
        CASE 
            WHEN TXN_DATE BETWEEN '2025-01-01' AND '2025-03-31' THEN 'Q1_2025'
            WHEN TXN_DATE BETWEEN '2025-04-01' AND '2025-06-30' THEN 'Q2_2025'
            WHEN TXN_DATE BETWEEN '2025-07-01' AND '2025-09-30' THEN 'Q3_2025'
            WHEN TXN_DATE >= '2025-10-01' THEN 'Q4_2025'
        END AS quarter,
        CASE 
            WHEN CATEGORY IN ('Travel', 'Dining') THEN 'Discretionary'
            WHEN CATEGORY IN ('Grocery', 'Utilities') THEN 'Essential'
            ELSE 'Other'
        END AS spending_type,
        AMOUNT
    FROM DEMO_RISK_DB.DEMO_RISK_DATA.PAYMENT_TRANSACTIONS
    WHERE STATUS = 'APPROVED' AND TXN_DATE >= '2025-01-01'
)
SELECT 
    quarter,
    ROUND(SUM(CASE WHEN spending_type = 'Discretionary' THEN AMOUNT ELSE 0 END), 0) AS discretionary,
    ROUND(SUM(CASE WHEN spending_type = 'Essential' THEN AMOUNT ELSE 0 END), 0) AS essential,
    ROUND(
        SUM(CASE WHEN spending_type = 'Discretionary' THEN AMOUNT ELSE 0 END) / 
        NULLIF(SUM(CASE WHEN spending_type = 'Essential' THEN AMOUNT ELSE 0 END), 0), 
    2) AS de_ratio
FROM categorized
WHERE quarter IS NOT NULL
GROUP BY quarter
ORDER BY quarter;

-- 4.2 Synthetic ID fraud surge analysis (pre vs post June 2025)
SELECT 
    CASE 
        WHEN INCIDENT_DATE < '2025-06-01' THEN 'Pre-Surge (Jan-May 2025)'
        ELSE 'Post-Surge (Jun-Nov 2025)'
    END AS period,
    COUNT(*) AS incident_count,
    ROUND(AVG(AMOUNT), 2) AS avg_amount,
    ROUND(SUM(AMOUNT), 0) AS total_amount
FROM DEMO_RISK_DB.DEMO_RISK_DATA.FRAUD_INCIDENTS
WHERE TYPE = 'Synthetic ID' AND INCIDENT_DATE >= '2025-01-01'
GROUP BY 1
ORDER BY 1;

-- 4.3 Fee sentiment trend (H1 vs H2 2025)
SELECT 
    CASE 
        WHEN SURVEY_DATE BETWEEN '2025-01-01' AND '2025-05-31' THEN 'H1 2025 (Jan-May)'
        ELSE 'H2 2025 (Jun-Nov)'
    END AS period,
    ROUND(AVG(SENTIMENT_SCORE), 2) AS avg_fee_sentiment,
    COUNT(*) AS response_count
FROM DEMO_RISK_DB.DEMO_RISK_DATA.CUSTOMER_SENTIMENT
WHERE TOPIC = 'Fees' AND SURVEY_DATE >= '2025-01-01'
GROUP BY 1
ORDER BY 1;
