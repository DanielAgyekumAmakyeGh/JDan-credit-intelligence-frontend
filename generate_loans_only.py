"""
Simple Loan Data Generator - Add loans to existing borrowers
"""

import pymysql
import random
from datetime import datetime, timedelta
from config.settings import DB_CONFIG

def main():
    print("=" * 60)
    print("XDSData Ghana - Loan Data Generator")
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
    if not lenders:
        # Insert default lenders
        default_lenders = [
            ('Absa Bank', 'regulated_bank'),
            ('GCB Bank', 'regulated_bank'),
            ('Stanbic Bank', 'regulated_bank'),
            ('MTN MoMo', 'fintech'),
            ('Telecel Cash', 'fintech'),
            ('Fido', 'fintech'),
            ('Bayport', 'microfinance')
        ]
        for name, ltype in default_lenders:
            cursor.execute("INSERT INTO lender (lender_name, lender_type) VALUES (%s, %s)", (name, ltype))
        conn.commit()
        cursor.execute("SELECT lender_id FROM lender")
        lenders = [row[0] for row in cursor.fetchall()]
    
    print(f"Found {len(lenders)} lenders")
    
    # Check current loan count
    cursor.execute("SELECT COUNT(*) FROM loan")
    current_loans = cursor.fetchone()[0]
    print(f"\nCurrent loans: {current_loans}")
    
    # Number of new loans to add
    target_loans = 2000
    loans_to_add = target_loans - current_loans
    
    if loans_to_add <= 0:
        print(f"✓ Already have {current_loans} loans (need {target_loans})")
        conn.close()
        return
    
    print(f"\n📊 Adding {loans_to_add} new loans...")
    print("-" * 40)
    
    statuses = ['active', 'paid', 'default']
    # 25% active, 60% paid, 15% default
    status_weights = [0.25, 0.60, 0.15]
    
    loans_added = 0
    for i in range(loans_to_add):
        borrower_id = random.choice(borrowers)
        lender_id = random.choice(lenders)
        amount = random.randint(500, 50000)
        days_offset = random.randint(0, 730)
        origination_date = datetime.now().date() - timedelta(days=days_offset)
        status = random.choices(statuses, weights=status_weights)[0]
        days_past_due = random.randint(15, 90) if status == 'default' else 0
        
        try:
            cursor.execute("""
                INSERT INTO loan (borrower_id, lender_id, original_amount, origination_date, loan_status, days_past_due)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (borrower_id, lender_id, amount, origination_date, status, days_past_due))
            loans_added += 1
            
            if loans_added % 500 == 0:
                print(f"  Added {loans_added} loans...")
                conn.commit()
                
        except Exception as e:
            pass
    
    conn.commit()
    
    # Verify new counts
    cursor.execute("SELECT COUNT(*) FROM loan")
    new_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM loan WHERE loan_status = 'default'")
    defaults = cursor.fetchone()[0]
    
    print(f"\n{'=' * 60}")
    print("DATA GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total loans: {new_count}")
    print(f"Defaults: {defaults} ({defaults/new_count*100:.1f}%)")
    print(f"\n✓ Ready for ML training!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()