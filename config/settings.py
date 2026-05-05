"""
Configuration settings for the XDSData Dashboard
"""

# Database settings - XAMPP MySQL with correct database name
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'xdsdata_ghana',  # <-- Changed to your actual database name
    'port': 3306
}

# Dashboard settings
DASHBOARD_CONFIG = {
    'title': 'XDSData Ghana - NPL Management Dashboard',
    'theme': 'light',
    'refresh_interval_seconds': 300,
    'default_months': 12,
    'npl_target_percent': 15.0,
    'high_risk_default_threshold': 20.0
}
