import pymysql

# Connect to MySQL
conn = pymysql.connect(host='localhost', user='root', password='', database='xdsdata_ghana')
cur = conn.cursor()

# Show all tables
cur.execute('SHOW TABLES')
tables = cur.fetchall()

print("=" * 50)
print("TABLES IN xdsdata_ghana DATABASE")
print("=" * 50)

for table in tables:
    table_name = table[0]
    print(f"\nTable: {table_name}")
    print("-" * 30)
    
    # Show columns for this table
    cur.execute(f'DESCRIBE {table_name}')
    columns = cur.fetchall()
    for col in columns:
        print(f"  {col[0]} : {col[1]}")
    
    # Show row count
    cur.execute(f'SELECT COUNT(*) FROM {table_name}')
    count = cur.fetchone()[0]
    print(f"  Total rows: {count}")

conn.close()

print("\n" + "=" * 50)
print("DATABASE CHECK COMPLETE")
print("=" * 50)
