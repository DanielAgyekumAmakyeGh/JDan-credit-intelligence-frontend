"""
Generate Additional Synthetic Data for ML Training
Based on actual xdsdata_ghana database schema
"""

import pymysql
import random
import hashlib
from datetime import datetime, timedelta
from config.settings import DB_CONFIG

def generate_borrower_data(cursor, num_borrowers=1000):
    """Generate additional borrower records based on actual schema"""
    
    first_names = ['Kwame', 'Akosua', 'Kofi', 'Ama', 'Yaw', 'Adwoa', 'Esi', 'Kojo', 
                   'Abena', 'Akwasi', 'Afia', 'Kwabena', 'Adjei', 'Mensah', 'Asare',
                   'Daniel', 'Michael', 'Sarah', 'Grace', 'James', 'Mary', 'John',
                   'David', 'Elizabeth', 'Joseph', 'Catherine', 'Francis', 'Patricia']
    last_names = ['Mensah', 'Osei', 'Asare', 'Adjei', 'Boateng', 'Opoku', 'Agyeman', 
                  'Frimpong', 'Tetteh', 'Sarfo', 'Appiah', 'Addy', 'Acquah', 'Annor',
                  'Twum', 'Poku', 'Boadu', 'Asiedu', 'Ampofo', 'Acheampong']
    borrower_types = ['individual', 'business']
    
    borrowers = []
    for i in range(num_borrowers):
        full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        ghana_card = f"GHA-{random.randint(100000000, 999999999)}"
        tin = f"TIN-{random.randint(100000, 999999)}"
        mobile = f"024{random.randint(1000000, 9999999)}"
        hashed_mobile = hashlib.sha256(mobile.encode()).hexdigest()
        borrower_type = random.choice(borrower_types)
        date_of_birth = datetime.now().date() - timedelta(days=random.randint(6570, 25550))
        registration_date = datetime.now().date() - timedelta(days=random.randint(0, 1095))
        
        try:
            cursor.execute("""
                INSERT INTO borrower (full_name, ghana_card, tin, hashed_mobile, borrower_type, date_of_birth, registration_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (full_name, ghana_card, tin, hashed_mobile, borrower_type, date_of_birth, registration_date))
            borrowers.append(cursor.lastrowid)
        except pymysql.err.IntegrityError:
            # Duplicate entry, skip
            pass
    
    return borrowers

def generate_loan_data(cursor, borrower_ids, num_loans=5000):
    """Generate loan records"""
    
    # Get existing lender IDs
    cursor.execute("SELECT lender_id FROM lender")
    lenders = [row[0] for row in cursor.fetchall()]
    if not lenders:
        # Insert some default lenders if none exist
        default_lenders = ['Absa Bank', 'GCB Bank', 'Stanbic Bank', 'MTN MoMo', 'Telecel Cash']
        for lender in default_lenders:
            cursor.execute("INSERT INTO lender (lender_name, lender_type) VALUES (%s, 'fintech')", (lender,))
        cursor.execute("SELECT lender_id FROM lender")
        lenders = [row[0] for row in cursor.fetchall()]
    
    statuses = ['active', 'paid', 'default']
    status_weights = [0.25, 0.60, 0.15]  # 15% default rate
    
    loans_added = 0
    for i in range(num_loans):
        borrower_id = random.choice(borrower_ids)
        lender_id = random.choice(lenders)
        amount = random.randint(500, 50000)
        origination_date = datetime.now().date() - timedelta(days=random.randint(0, 730))
        status = random.choices(statuses, weights=status_weights)[0]
        days_past_due = random.randint(0, 90) if status == 'default' else 0
        
        try:
            cursor.execute("""
                INSERT INTO loan (borrower_id, lender_id, original_amount, origination_date, loan_status, days_past_due)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (borrower_id, lender_id, amount, origination_date, status, days_past_due))
            loans_added += 1
        except Exception as e:
            pass
    
    return loans_added

def generate_credit_scores(cursor, borrower_ids):
    """Generate credit scores for borrowers"""
    
    scores_added = 0
    for borrower_id in borrower_ids:
        credit_score = random.randint(450, 800)
        default_prob = round(1 - (credit_score - 300) / 550, 4)
        default_prob = max(0.05, min(0.95, default_prob))
        score_date = datetime.now().date() - timedelta(days=random.randint(0, 90))
        
        try:
            cursor.execute("""
                INSERT INTO credit_score (borrower_id, credit_score, default_probability, score_date, model_version)
                VALUES (%s, %s, %s, %s, 'v1.0')
            """, (borrower_id, credit_score, default_prob, score_date))
            scores_added += 1
        except Exception as e:
            pass
    
    return scores_added

def generate_mobile_summary(cursor, borrower_ids):
    """Generate mobile money summary data"""
    
    summaries_added = 0
    for borrower_id in borrower_ids:
        avg_volume = random.randint(500, 20000)
        transaction_freq = round(random.uniform(5, 30), 2)
        airtime_consistency = round(random.uniform(0.3, 0.95), 2)
        
        try:
            cursor.execute("""
                INSERT INTO borrower_mobile_summary (borrower_id, avg_monthly_volume, avg_transaction_frequency, avg_airtime_topup_consistency)
                VALUES (%s, %s, %s, %s)
            """, (borrower_id, avg_volume, transaction_freq, airtime_consistency))
            summaries_added += 1
        except Exception as e:
            pass
    
    return summaries_added

def generate_credit_summary(cursor, borrower_ids):
    """Generate credit summary data"""
    
    summaries_added = 0
    for borrower_id in borrower_ids:
        active_loans = random.randint(0, 3)
        total_debt = random.randint(0, 50000)
        past_defaults = random.randint(0, 2)
        max_days_past_due = random.randint(0, 60) if past_defaults > 0 else 0
        months_since_default = random.randint(6, 60) if past_defaults > 0 else 999
        
        try:
            cursor.execute("""
                INSERT INTO borrower_credit_summary (borrower_id, num_active_loans, total_outstanding_balance, num_past_defaults, max_days_past_due, months_since_last_default)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (borrower_id, active_loans, total_debt, past_defaults, max_days_past_due, months_since_default))
            summaries_added += 1
        except Exception as e:
            pass
    
    return summaries_added

def generate_loan_applications(cursor, borrower_ids, num_apps=3000):
    """Generate loan application records"""
    
    purposes = ['Business', 'Emergency', 'Education', 'Housing', 'Medical', 'Personal', 'Agriculture']
    statuses = ['approved', 'rejected', 'pending']
    status_weights = [0.70, 0.15, 0.15]
    
    apps_added = 0
    for i in range(num_apps):
        borrower_id = random.choice(borrower_ids)
        purpose = random.choice(purposes)
        amount = random.randint(500, 50000)
        tenor = random.choice([3, 6, 12, 18, 24])
        status = random.choices(statuses, weights=status_weights)[0]
        app_date = datetime.now().date() - timedelta(days=random.randint(0, 180))
        
        try:
            cursor.execute("""
                INSERT INTO loan_application (borrower_id, loan_purpose, requested_amount, requested_tenor, application_status, application_date)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (borrower_id, purpose, amount, tenor, status, app_date))
            apps_added += 1
        except Exception as e:
            pass
    
    return apps_added

def main():
    print("=" * 60)
    print("XDSData Ghana - Synthetic Data Generator")
    print("=" * 60)
    
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Connected to database")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    # Check current counts
    cursor.execute("SELECT COUNT(*) FROM loan")
    current_loans = cursor.fetchone()[0]
    print(f"\nCurrent loans: {current_loans}")
    
    if current_loans >= 1000:
        print("✓ Already have sufficient data (1000+ records)")
        response = input("Do you want to generate more data anyway? (y/n): ")
        if response.lower() != 'y':
            conn.close()
            return
    
    # Generate data
    print("\n📊 Generating synthetic data...")
    print("-" * 40)
    
    # 1. Generate borrowers
    print("  Generating borrowers...", end=" ", flush=True)
    borrower_ids = generate_borrower_data(cursor, num_borrowers=800)
    conn.commit()
    print(f"✓ {len(borrower_ids)} added")
    
    # 2. Generate loans
    print("  Generating loans...", end=" ", flush=True)
    loans_added = generate_loan_data(cursor, borrower_ids, num_loans=3000)
    conn.commit()
    print(f"✓ {loans_added} added")
    
    # 3. Generate credit scores
    print("  Generating credit scores...", end=" ", flush=True)
    scores_added = generate_credit_scores(cursor, borrower_ids)
    conn.commit()
    print(f"✓ {scores_added} added")
    
    # 4. Generate mobile summaries
    print("  Generating mobile summaries...", end=" ", flush=True)
    mobile_added = generate_mobile_summary(cursor, borrower_ids)
    conn.commit()
    print(f"✓ {mobile_added} added")
    
    # 5. Generate credit summaries
    print("  Generating credit summaries...", end=" ", flush=True)
    credit_added = generate_credit_summary(cursor, borrower_ids)
    conn.commit()
    print(f"✓ {credit_added} added")
    
    # 6. Generate loan applications
    print("  Generating loan applications...", end=" ", flush=True)
    apps_added = generate_loan_applications(cursor, borrower_ids, num_apps=2000)
    conn.commit()
    print(f"✓ {apps_added} added")
    
    # Verify final counts
    cursor.execute("SELECT COUNT(*) FROM borrower")
    total_borrowers = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM loan")
    total_loans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM loan WHERE loan_status = 'default'")
    total_defaults = cursor.fetchone()[0]
    default_rate = (total_defaults / total_loans * 100) if total_loans > 0 else 0
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total borrowers: {total_borrowers}")
    print(f"Total loans: {total_loans}")
    print(f"Defaults: {total_defaults} ({default_rate:.1f}%)")
    print("\n✓ Ready for ML training!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()