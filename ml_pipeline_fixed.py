"""
Fixed ML Pipeline - Converts Decimal to Float
Pure PyMySQL Version - Works with Python 3.14
"""

import pymysql
import pandas as pd
import numpy as np
from decimal import Decimal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

def convert_decimal_to_float(df):
    """Convert Decimal columns to float"""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if first non-null value is Decimal
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample, Decimal):
                df[col] = df[col].astype(float)
                print(f"  Converted {col} from Decimal to float")
    return df

def fetch_data():
    """Fetch data using pure PyMySQL (no SQLAlchemy)"""
    
    print("=" * 70)
    print("FETCHING DATA FROM DATABASE")
    print("=" * 70)
    
    conn = pymysql.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        CAST(l.original_amount AS DECIMAL(10,2)) AS original_amount,
        l.days_past_due,
        l.interest_rate,
        l.tenor_months,
        b.borrower_type,
        TIMESTAMPDIFF(YEAR, b.date_of_birth, CURDATE()) AS age,
        DATEDIFF(CURDATE(), b.registration_date) AS days_as_customer,
        COALESCE(bc.num_active_loans, 0) AS active_loans,
        COALESCE(bc.total_outstanding_balance, 0) AS total_debt,
        COALESCE(bc.num_loans_past_due, 0) AS loans_past_due,
        COALESCE(bc.max_days_past_due, 0) AS max_days_past_due,
        COALESCE(bc.num_past_defaults, 0) AS past_defaults,
        COALESCE(bc.months_since_last_default, 999) AS months_since_default,
        COALESCE(bm.avg_monthly_volume, 0) AS monthly_income,
        COALESCE(bm.avg_transaction_frequency, 0) AS transaction_frequency,
        COALESCE(bm.avg_airtime_topup_consistency, 0) AS airtime_consistency,
        COALESCE(cs.credit_score, 600) AS credit_score,
        COALESCE(bf.night_applications, 0) AS night_applications,
        COALESCE(bf.total_applications_last_30_days, 0) AS recent_applications,
        COALESCE((
            SELECT AVG(CASE WHEN bill_status = 'paid' THEN 1 ELSE 0 END)
            FROM utility_bill ub WHERE ub.borrower_id = b.borrower_id
        ), 0) AS utility_score,
        COALESCE(la.loan_purpose, 'Unknown') AS loan_purpose,
        CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END AS default_flag
    FROM loan l
    JOIN borrower b ON l.borrower_id = b.borrower_id
    LEFT JOIN borrower_credit_summary bc ON b.borrower_id = bc.borrower_id
    LEFT JOIN borrower_mobile_summary bm ON b.borrower_id = bm.borrower_id
    LEFT JOIN credit_score cs ON b.borrower_id = cs.borrower_id
    LEFT JOIN borrower_features_all bf ON b.borrower_id = bf.borrower_id
    LEFT JOIN loan_application la ON l.borrower_id = la.borrower_id
    WHERE l.loan_status IN ('paid', 'default')
    """
    
    print("Executing query...")
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Fetch column names
    columns = [desc[0] for desc in cursor.description]
    
    # Fetch all rows
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Convert Decimal to float
    df = convert_decimal_to_float(df)
    
    # Ensure all numeric columns are float
    for col in df.columns:
        if col not in ['borrower_type', 'loan_purpose', 'default_flag']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                pass
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Default rate: {df['default_flag'].mean()*100:.1f}%")
    
    return df

def engineer_features(df):
    """Create derived features"""
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    # Ensure all values are float
    for col in ['total_debt', 'monthly_income', 'credit_score', 'original_amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Debt-to-Income Ratio
    denominator = df['monthly_income'] * 12 + 1
    df['dti_ratio'] = df['total_debt'] / denominator
    df['dti_ratio'] = df['dti_ratio'].clip(0, 2)
    
    # Monthly debt service
    df['monthly_debt'] = df['total_debt'] / 12
    df['disposable_income'] = (df['monthly_income'] * 0.3) - df['monthly_debt']
    df['disposable_income'] = df['disposable_income'].clip(0, None)
    
    # Default severity
    df['default_severity'] = df['past_defaults'] * (df['max_days_past_due'] / 30)
    
    # Income stability
    df['income_stability'] = df['transaction_frequency'] * df['airtime_consistency']
    
    # Risk score (inverse of credit score)
    df['risk_score'] = 850 - df['credit_score']
    
    # Loan-to-Income ratio
    df['loan_to_income'] = df['original_amount'] / (df['monthly_income'] + 1)
    df['loan_to_income'] = df['loan_to_income'].clip(0, 5)
    
    # Recent distress indicator
    df['recent_distress'] = ((df['months_since_default'] < 12) & (df['past_defaults'] > 0)).astype(int)
    
    # Night application risk
    df['night_risk'] = df['night_applications'] * 0.1
    
    # Tenor risk
    df['tenor_risk'] = df['tenor_months'] / 24
    
    # Combined risk score
    df['combined_risk'] = (
        df['risk_score'] * 0.3 +
        df['dti_ratio'] * 100 * 0.25 +
        df['past_defaults'] * 50 * 0.2 +
        df['default_severity'] * 0.15 +
        df['tenor_risk'] * 100 * 0.1
    )
    
    print(f"✓ Created {len(df.columns)} total features")
    
    return df

def clean_data(df):
    """Handle missing values and outliers"""
    
    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"✓ Removed {before - len(df)} duplicate rows")
    
    # Handle missing values - convert to numeric first
    for col in df.columns:
        if col not in ['default_flag']:
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN with median or 0
            if df[col].isnull().sum() > 0:
                if col in ['borrower_type', 'loan_purpose']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median() if df[col].median() is not None else 0)
    
    # Cap outliers (using 99th percentile)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'default_flag':
            upper = df[col].quantile(0.99)
            lower = df[col].quantile(0.01)
            if upper > lower:
                df[col] = df[col].clip(lower, upper)
    
    print(f"✓ Processed {len(numeric_cols)} numeric columns")
    
    return df

def encode_categorical(df):
    """Encode categorical variables"""
    
    print("\n" + "=" * 70)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  ✓ {col}: encoded")
    
    return df

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate models"""
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    # Calculate class weight
    default_rate = y_train.mean()
    if default_rate > 0:
        class_weight = {0: 1, 1: (1 - default_rate) / default_rate}
    else:
        class_weight = {0: 1, 1: 1}
    
    print(f"✓ Default rate: {default_rate*100:.1f}%")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=(1 - default_rate) / max(default_rate, 0.01),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {'model': model, 'auc': auc}
        print(f"  {name}: AUC = {auc:.4f}")
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', models['Random Forest']), ('xgb', models['XGBoost'])],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    ensemble_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test_scaled)[:, 1])
    print(f"  Ensemble: AUC = {ensemble_auc:.4f}")
    
    # Select best model
    if ensemble_auc > max(results[r]['auc'] for r in results):
        best_model = ensemble
        best_auc = ensemble_auc
        best_name = "Ensemble"
    else:
        best_name = max(results, key=lambda x: results[x]['auc'])
        best_model = results[best_name]['model']
        best_auc = results[best_name]['auc']
    
    # Find optimal threshold
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
    best_threshold = 0.5
    if len(thresholds) > 0 and len(f1_scores) > 0:
        best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Final evaluation
    y_pred = (y_proba >= best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': best_auc,
        'best_threshold': best_threshold,
        'best_model': best_name,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'default_rate': default_rate
    }
    
    return best_model, scaler, metrics

def main():
    print("\n" + "=" * 70)
    print("ML PIPELINE FOR CREDIT DEFAULT PREDICTION")
    print("Fixed Version - Decimal Conversion")
    print("=" * 70)
    
    # Fetch data
    df_raw = fetch_data()
    
    if len(df_raw) < 100:
        print(f"\n❌ Error: Need at least 100 records. Found {len(df_raw)}")
        return
    
    # Engineer features
    df = engineer_features(df_raw)
    
    # Clean data
    df = clean_data(df)
    
    # Encode categorical
    df = encode_categorical(df)
    
    # Separate features and target
    y = df['default_flag']
    X = df.drop(columns=['default_flag'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {len(X_train)} records")
    print(f"  Test: {len(X_test)} records")
    
    # Train models
    best_model, scaler, metrics = train_models(X_train, y_train, X_test, y_test)
    
    # Save model
    joblib.dump(best_model, 'models/credit_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Final report
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FINAL REPORT")
    print("=" * 70)
    print(f"\n📊 Best Model: {metrics['best_model']}")
    print(f"📈 Optimal Threshold: {metrics['best_threshold']:.3f}")
    print(f"\n🎯 Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1 Score:  {metrics['f1']:.3f}")
    print(f"   AUC-ROC:   {metrics['auc']:.3f}")
    
    print(f"\n📊 Data Summary:")
    print(f"   Training samples: {metrics['training_samples']}")
    print(f"   Test samples: {metrics['test_samples']}")
    print(f"   Default rate: {metrics['default_rate']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ Model saved to 'models/credit_model.pkl'")
    print("=" * 70)

if __name__ == "__main__":
    from config.settings import DB_CONFIG
    main()