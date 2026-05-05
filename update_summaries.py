"""
Update borrower credit summaries based on loan data
"""

import pymysql
from config.settings import DB_CONFIG

def main():
    print("=" * 60)
    print("Updating Borrower Credit Summaries")
    print("=" * 60)
    
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Connected to database")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    # First, check current summary counts
    cursor.execute("SELECT COUNT(*) FROM borrower_credit_summary")
    before_count = cursor.fetchone()[0]
    print(f"\nCurrent credit summaries: {before_count}")
    
    # Update existing summaries
    print("\nUpdating existing summaries...")
    cursor.execute("""
        UPDATE borrower_credit_summary bc
        SET 
            num_active_loans = (
                SELECT COALESCE(COUNT(*), 0) FROM loan l 
                WHERE l.borrower_id = bc.borrower_id AND l.loan_status = 'active'
            ),
            total_outstanding_balance = (
                SELECT COALESCE(SUM(l.original_amount), 0) FROM loan l 
                WHERE l.borrower_id = bc.borrower_id AND l.loan_status = 'active'
            ),
            num_loans_past_due = (
                SELECT COALESCE(COUNT(*), 0) FROM loan l 
                WHERE l.borrower_id = bc.borrower_id AND l.days_past_due > 30
            ),
            num_past_defaults = (
                SELECT COALESCE(COUNT(*), 0) FROM loan l 
                WHERE l.borrower_id = bc.borrower_id AND l.loan_status = 'default'
            ),
            max_days_past_due = (
                SELECT COALESCE(MAX(l.days_past_due), 0) FROM loan l 
                WHERE l.borrower_id = bc.borrower_id AND l.loan_status = 'default'
            ),
            months_since_last_default = (
                SELECT COALESCE(
                    TIMESTAMPDIFF(MONTH, MAX(l.origination_date), CURDATE()), 
                    999
                ) FROM loan l 
                WHERE l.borrower_id = bc.borrower_id AND l.loan_status = 'default'
            )
    """)
    
    updated_rows = cursor.rowcount
    print(f"  Updated {updated_rows} summaries")
    
    # Insert summaries for borrowers without one
    print("\nInserting missing summaries...")
    cursor.execute("""
        INSERT INTO borrower_credit_summary (
            borrower_id, 
            num_active_loans, 
            total_outstanding_balance, 
            num_loans_past_due,
            num_past_defaults, 
            max_days_past_due, 
            months_since_last_default
        )
        SELECT 
            b.borrower_id,
            COALESCE((SELECT COUNT(*) FROM loan l WHERE l.borrower_id = b.borrower_id AND l.loan_status = 'active'), 0),
            COALESCE((SELECT SUM(l.original_amount) FROM loan l WHERE l.borrower_id = b.borrower_id AND l.loan_status = 'active'), 0),
            COALESCE((SELECT COUNT(*) FROM loan l WHERE l.borrower_id = b.borrower_id AND l.days_past_due > 30), 0),
            COALESCE((SELECT COUNT(*) FROM loan l WHERE l.borrower_id = b.borrower_id AND l.loan_status = 'default'), 0),
            COALESCE((SELECT MAX(l.days_past_due) FROM loan l WHERE l.borrower_id = b.borrower_id AND l.loan_status = 'default'), 0),
            COALESCE(
                (SELECT TIMESTAMPDIFF(MONTH, MAX(l.origination_date), CURDATE()) 
                 FROM loan l WHERE l.borrower_id = b.borrower_id AND l.loan_status = 'default'), 
                999
            )
        FROM borrower b
        LEFT JOIN borrower_credit_summary bc ON b.borrower_id = bc.borrower_id
        WHERE bc.borrower_id IS NULL
    """)
    
    inserted_rows = cursor.rowcount
    print(f"  Inserted {inserted_rows} new summaries")
    
    conn.commit()
    
    # Verify final counts
    cursor.execute("SELECT COUNT(*) FROM borrower_credit_summary")
    after_count = cursor.fetchone()[0]
    
    # Show summary statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_borrowers,
            SUM(num_active_loans) as total_active_loans,
            SUM(num_past_defaults) as total_defaults,
            AVG(num_active_loans) as avg_active_loans
        FROM borrower_credit_summary
    """)
    stats = cursor.fetchone()
    
    print(f"\n{'=' * 60}")
    print("UPDATE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total credit summaries: {after_count}")
    print(f"Total active loans: {stats[1]}")
    print(f"Total defaults (historical): {stats[2]}")
    print(f"Avg active loans per borrower: {stats[3]:.2f}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()