-- =====================================================
-- XDSData Ghana - Database Setup Script
-- Run this script to create all required tables and views
-- =====================================================

-- Create database
CREATE DATABASE IF NOT EXISTS xds_credit_bureau;
USE xds_credit_bureau;

-- 1. Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    date_of_birth DATE,
    ghana_card VARCHAR(20) UNIQUE,
    region VARCHAR(100),
    registration_date DATE DEFAULT CURRENT_DATE,
    INDEX idx_ghana_card (ghana_card)
);

-- 2. Lenders table
CREATE TABLE IF NOT EXISTS lender (
    lender_id INT AUTO_INCREMENT PRIMARY KEY,
    lender_name VARCHAR(255) NOT NULL,
    lender_type ENUM('regulated_bank', 'microfinance', 'fintech', 'sacco') DEFAULT 'fintech',
    registration_date DATE,
    data_sharing_agreement BOOLEAN DEFAULT FALSE
);

-- 3. Loans table
CREATE TABLE IF NOT EXISTS loan (
    loan_id INT AUTO_INCREMENT PRIMARY KEY,
    borrower_id INT NOT NULL,
    lender_id INT NOT NULL,
    original_amount DECIMAL(15,2),
    origination_date DATE,
    tenor_months INT DEFAULT 12,
    loan_status ENUM('active', 'paid', 'default') DEFAULT 'active',
    past_due_days INT DEFAULT 0,
    FOREIGN KEY (borrower_id) REFERENCES customers(customer_id),
    FOREIGN KEY (lender_id) REFERENCES lender(lender_id),
    INDEX idx_borrower_date (borrower_id, origination_date),
    INDEX idx_status (loan_status)
);

-- 4. Loan applications table
CREATE TABLE IF NOT EXISTS loan_application (
    application_id INT AUTO_INCREMENT PRIMARY KEY,
    borrower_id INT NOT NULL,
    lender_id INT NOT NULL,
    loan_purpose VARCHAR(100),
    application_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
    application_date DATE,
    FOREIGN KEY (borrower_id) REFERENCES customers(customer_id),
    FOREIGN KEY (lender_id) REFERENCES lender(lender_id)
);

-- 5. Alternative data table
CREATE TABLE IF NOT EXISTS alternative_data (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    momo_avg_monthly_vol DECIMAL(15,2),
    momo_active_days_per_month INT,
    utility_payment_score INT CHECK (utility_payment_score BETWEEN 0 AND 2),
    airtime_recharges_per_week INT,
    data_month DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 6. Mobile money transactions table
CREATE TABLE IF NOT EXISTS mobile_money_transaction (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    hashed_mobile VARCHAR(255),
    borrower_id INT,
    transaction_date DATE,
    transaction_type ENUM('send', 'receive', 'pay_bill'),
    amount DECIMAL(15,2),
    FOREIGN KEY (borrower_id) REFERENCES customers(customer_id),
    INDEX idx_borrower (borrower_id)
);

-- 7. Borrower features view (for risk scoring)
CREATE OR REPLACE VIEW borrower_features_all AS
SELECT 
    c.customer_id AS borrower_id,
    c.full_name,
    c.ghana_card,
    COALESCE(COUNT(DISTINCT l.loan_id), 0) AS num_active_loans,
    COALESCE(SUM(l.original_amount), 0) AS total_outstanding_balance,
    COALESCE(SUM(CASE WHEN l.past_due_days > 30 THEN 1 ELSE 0 END), 0) AS num_loans_past_due,
    COALESCE(MAX(l.past_due_days), 0) AS max_days_past_due,
    COALESCE(SUM(CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END), 0) AS num_past_defaults,
    COALESCE(a.momo_avg_monthly_vol, 0) AS avg_monthly_volume,
    COALESCE(a.utility_payment_score, 0) AS elec_on_time_count,
    COALESCE(a.momo_active_days_per_month, 0) AS night_applications,
    c.registration_date AS last_activity_date
FROM customers c
LEFT JOIN loan l ON c.customer_id = l.borrower_id AND l.loan_status IN ('active', 'default')
LEFT JOIN alternative_data a ON c.customer_id = a.customer_id
GROUP BY c.customer_id;

-- Create indexes for performance
CREATE INDEX idx_loan_origination ON loan(origination_date);
CREATE INDEX idx_loan_borrower_status ON loan(borrower_id, loan_status);
CREATE INDEX idx_momo_date ON mobile_money_transaction(transaction_date);

-- Grant privileges (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON xds_credit_bureau.* TO 'dashboard_user'@'localhost' IDENTIFIED BY 'secure_password';
-- FLUSH PRIVILEGES;