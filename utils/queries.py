"""
SQL Queries for XDSData Dashboard
Updated to match actual xdsdata_ghana database schema
"""

# NPL Trend Query - Using loan table
NPL_TREND_QUERY = """
SELECT
    DATE_FORMAT(origination_date, '%%Y-%%m') AS month,
    SUM(original_amount) AS total_disbursed,
    SUM(CASE WHEN loan_status = 'default' THEN original_amount ELSE 0 END) AS defaulted_amount,
    ROUND(100 * SUM(CASE WHEN loan_status = 'default' THEN original_amount ELSE 0 END) / 
          NULLIF(SUM(original_amount), 0), 2) AS npl_ratio
FROM loan
WHERE origination_date >= DATE_SUB(CURDATE(), INTERVAL %s MONTH)
GROUP BY DATE_FORMAT(origination_date, '%%Y-%%m')
ORDER BY month
"""

# Lender Performance Query
LENDER_PERFORMANCE_QUERY = """
SELECT
    l.lender_name,
    l.lender_type,
    COUNT(ln.loan_id) AS total_loans,
    SUM(CASE WHEN ln.loan_status = 'default' THEN 1 ELSE 0 END) AS defaults,
    ROUND(100 * SUM(CASE WHEN ln.loan_status = 'default' THEN 1 ELSE 0 END) / 
          NULLIF(COUNT(ln.loan_id), 0), 2) AS default_rate
FROM lender l
JOIN loan ln ON l.lender_id = ln.lender_id
WHERE ln.origination_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
GROUP BY l.lender_id
ORDER BY default_rate DESC
"""

# Loan Purpose Default Query
LOAN_PURPOSE_QUERY = """
SELECT
    la.loan_purpose,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END) AS defaults,
    ROUND(100 * SUM(CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END) / 
          NULLIF(COUNT(*), 0), 1) AS default_rate_pct
FROM loan_application la
JOIN loan l ON la.borrower_id = l.borrower_id AND la.lender_id = l.lender_id
WHERE la.application_status = 'approved'
GROUP BY la.loan_purpose
ORDER BY default_rate_pct DESC
"""

# Loan Stacking Check Query
LOAN_STACKING_QUERY = """
SELECT COUNT(*) AS loans_today
FROM loan
WHERE borrower_id = %s AND origination_date = CURDATE()
"""

# Summary Statistics Query
SUMMARY_STATS_QUERY = """
SELECT
    (SELECT COUNT(*) FROM loan) AS total_loans,
    (SELECT COUNT(*) FROM loan WHERE loan_status = 'default') AS total_defaults,
    (SELECT ROUND(AVG(original_amount), 0) FROM loan) AS avg_loan_amount,
    (SELECT COUNT(*) FROM loan WHERE origination_date >= CURDATE() - INTERVAL 30 DAY) AS loans_last_30d,
    (SELECT COUNT(*) FROM loan WHERE loan_status = 'default' AND origination_date >= CURDATE() - INTERVAL 30 DAY) AS defaults_last_30d
"""

# Borrower Risk Profile Query - Using borrower_features_all view
BORROWER_RISK_QUERY = """
SELECT
    borrower_id,
    num_active_loans,
    total_outstanding_balance,
    num_loans_past_due,
    max_days_past_due,
    avg_monthly_volume AS income_proxy,
    elec_on_time_count
FROM borrower_features_all
WHERE ghana_card = %s
"""

# Get top 5 borrowers by score
TOP_BORROWERS_QUERY = """
SELECT
    b.full_name,
    b.ghana_card,
    cs.credit_score,
    cs.default_probability,
    cs.score_date
FROM credit_score cs
JOIN borrower b ON cs.borrower_id = b.borrower_id
WHERE cs.score_date = (SELECT MAX(score_date) FROM credit_score)
ORDER BY cs.credit_score DESC
LIMIT 10
"""

# Get high risk borrowers
HIGH_RISK_BORROWERS_QUERY = """
SELECT
    b.full_name,
    b.ghana_card,
    cs.credit_score,
    cs.default_probability,
    bc.num_active_loans,
    bc.total_outstanding_balance
FROM credit_score cs
JOIN borrower b ON cs.borrower_id = b.borrower_id
JOIN borrower_credit_summary bc ON b.borrower_id = bc.borrower_id
WHERE cs.credit_score < 500
ORDER BY cs.credit_score ASC
LIMIT 10
"""

# Recent loans with status
RECENT_LOANS_QUERY = """
SELECT
    l.loan_id,
    b.full_name AS borrower_name,
    lnd.lender_name,
    l.original_amount,
    l.origination_date,
    l.loan_status,
    l.days_past_due
FROM loan l
JOIN borrower b ON l.borrower_id = b.borrower_id
JOIN lender lnd ON l.lender_id = lnd.lender_id
ORDER BY l.origination_date DESC
LIMIT 20
"""
