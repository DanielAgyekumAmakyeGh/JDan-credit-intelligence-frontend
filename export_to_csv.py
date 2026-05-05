"""
Ultra-Simple Database to CSV Exporter
No pandas, no SQLAlchemy - just pure Python and CSV
"""

import pymysql
import csv
import os
from datetime import datetime
from config.settings import DB_CONFIG

def export_table(cursor, table_name, output_dir):
    """Export a single table to CSV"""
    try:
        # Get column names
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = [row[0] for row in cursor.fetchall()]
        
        # Get all data
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        # Write to CSV
        filepath = os.path.join(output_dir, f"{table_name}.csv")
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        
        return len(rows)
    except Exception as e:
        print(f"  ERROR: {table_name} - {e}")
        return -1

def main():
    print("=" * 50)
    print("XDSData Ghana - Simple CSV Exporter")
    print("=" * 50)
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"csv_export_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput folder: {output_dir}")
    
    # Connect to database
    print("\nConnecting to database...")
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    # Get all tables
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nFound {len(tables)} tables")
    
    # Export each table
    print("\nExporting tables...")
    print("-" * 30)
    
    success_count = 0
    for table in tables:
        print(f"  {table}...", end=" ", flush=True)
        row_count = export_table(cursor, table, output_dir)
        if row_count >= 0:
            print(f"✓ {row_count} rows")
            success_count += 1
        else:
            print(f"✗ Failed")
    
    # Close connection
    cursor.close()
    conn.close()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"EXPORT COMPLETE")
    print(f"Folder: {output_dir}")
    print(f"Path: C:\\Users\\USER\\xdsdata\\{output_dir}")
    print(f"Tables exported: {success_count} of {len(tables)}")
    print("=" * 50)

if __name__ == "__main__":
    main()