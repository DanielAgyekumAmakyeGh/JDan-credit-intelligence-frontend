"""
Check data counts in database
"""

import pymysql
from config.settings import DB_CONFIG

conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()

print("=" * 50)
print("XDSData Ghana - Database Data Counts")
print("=" * 50)

# Check all tables
tables = ['borrower', 'loan', 'lender', 'credit_score', 
          'borrower_mobile_summary', 'borrower_credit_summary', 
          'loan_application', 'borrower_features_all']

for table in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table:30} : {count:6} records")
    except Exception as e:
        print(f"{table:30} : ERROR - {str(e)[:50]}")

# Check loan status breakdown
print("\n" + "-" * 50)
print("Loan Status Breakdown:")
cursor.execute("SELECT loan_status, COUNT(*) FROM loan GROUP BY loan_status")
for row in cursor.fetchall():
    print(f"  {row[0]:10} : {row[1]}")

# Check default rate
cursor.execute("SELECT COUNT(*) FROM loan WHERE loan_status = 'default'")
defaults = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM loan")
total = cursor.fetchone()[0]
print(f"\nDefault Rate: {defaults}/{total} = {defaults/total*100:.1f}%")

conn.close()