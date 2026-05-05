"""
Debugged Loan Data Generator - Shows actual errors
"""

import pymysql
import random
from datetime import datetime, timedelta
from config.settings import DB_CONFIG

def main():
    print("=" * 60)
    print("XDSData Ghana - Loan Data Generator (Debug)")
    print("=" * 60)
    
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Connected to database")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    # Get existing borrowers
    cursor.execute("SELECT borrower_id FROM borrower")
    borrowers = [row[0] for row in cursor.fetchall()]
    print(f"\nFound {len(borrowers)} borrowers")
    
    # Get existing lenders
    cursor.execute("SELECT lender_id FROM lender")
    lenders = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(lenders)} lenders")
    
    # Check current loan count
    cursor.execute("SELECT COUNT(*) FROM loan")
    current_loans = cursor.fetchone()[0]
    print(f"\nCurrent loans: {current_loans}")
    
    # Target
    target_loans = 2000
    loans_to_add = target_loans - current_loans
    
    if loans_to_add <= 0:
        print(f"✓ Already have {current_loans} loans")
        conn.close()
        return
    
    print(f"\n📊 Attempting to add {loans_to_add} loans...")
    print("-" * 40)
    
    statuses = ['active', 'paid', 'default']
    status_weights = [0.25, 0.60, 0.15]
    
    # Check table structure first
    print("\nChecking loan table structure...")
    cursor.execute("DESCRIBE loan")
    columns = [row[0] for row in cursor.fetchall()]
    print(f"Columns in loan table: {columns}")
    
    loans_added = 0
    errors = 0
    
    for i in range(loans_to_add):
        borrower_id = random.choice(borrowers)
        lender_id = random.choice(lenders)
        amount = random.randint(500, 50000)
        days_offset = random.randint(0, 730)
        origination_date = datetime.now().date() - timedelta(days=days_offset)
        status = random.choices(statuses, weights=status_weights)[0]
        days_past_due = random.randint(15, 90) if status == 'default' else 0
        
        try:
            # Try different INSERT statements based on actual columns
            if 'days_past_due' in columns:
                cursor.execute("""
                    INSERT INTO loan (borrower_id, lender_id, original_amount, origination_date, loan_status, days_past_due)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (borrower_id, lender_id, amount, origination_date, status, days_past_due))
            else:
                cursor.execute("""
                    INSERT INTO loan (borrower_id, lender_id, original_amount, origination_date, loan_status)
                    VALUES (%s, %s, %s, %s, %s)
                """, (borrower_id, lender_id, amount, origination_date, status))
            
            loans_added += 1
            
            if loans_added % 200 == 0:
                conn.commit()
                print(f"  Added {loans_added} loans...")
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {e}")
    
    conn.commit()
    
    # Verify new counts
    cursor.execute("SELECT COUNT(*) FROM loan")
    new_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM loan WHERE loan_status = 'default'")
    defaults = cursor.fetchone()[0]
    
    print(f"\n{'=' * 60}")
    print("DATA GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Loans attempted: {loans_to_add}")
    print(f"Loans added: {loans_added}")
    print(f"Errors: {errors}")
    print(f"Total loans now: {new_count}")
    print(f"Defaults: {defaults} ({defaults/new_count*100:.1f}%)" if new_count > 0 else "No loans")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()