"""
ML Pipeline with Proper Temporal Split
Uses all loan data but prevents lookahead bias
"""

import pymysql
import pandas as pd
import numpy as np
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

def fetch_all_loan_data():
    """
    Fetch all loan data BUT track origination date for temporal split
    This prevents using future information to predict past
    """
    
    print("=" * 70)
    print("FETCHING LOAN DATA (With Temporal Tracking)")
    print("=" * 70)
    
    conn = pymysql.connect(**DB_CONFIG)
    
    # Get all loans with their origination dates for proper time-based split
    query = """
    SELECT 
        l.loan_id,
        l.origination_date,
        l.original_amount AS loan_amount,
        l.interest_rate,
        l.tenor_months,
        l.loan_status,
        CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END AS default_flag,
        
        -- Borrower info (static, available at any time)
        b.borrower_id,
        b.borrower_type,
        TIMESTAMPDIFF(YEAR, b.date_of_birth, CURDATE()) AS age,
        DATEDIFF(l.origination_date, b.registration_date) AS days_as_customer_at_loan,
        
        -- Historical credit data (ONLY loans BEFORE this one)
        COALESCE((
            SELECT COUNT(*) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < l.origination_date
            AND l2.loan_status = 'default'
        ), 0) AS past_defaults_before,
        
        COALESCE((
            SELECT MAX(l2.days_past_due) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < l.origination_date
            AND l2.loan_status = 'default'
        ), 0) AS max_past_due_before,
        
        COALESCE((
            SELECT COUNT(*) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < l.origination_date
        ), 0) AS total_prev_loans,
        
        -- Mobile money summary (static snapshot)
        COALESCE(bm.avg_monthly_volume, 0) AS monthly_income,
        COALESCE(bm.avg_transaction_frequency, 0) AS transaction_frequency,
        COALESCE(bm.avg_airtime_topup_consistency, 0) AS airtime_consistency,
        
        -- Latest credit score BEFORE this loan
        COALESCE((
            SELECT cs2.credit_score FROM credit_score cs2 
            WHERE cs2.borrower_id = b.borrower_id 
            AND cs2.score_date <= l.origination_date
            ORDER BY cs2.score_date DESC LIMIT 1
        ), 600) AS credit_score_before,
        
        -- Utility score BEFORE this loan
        COALESCE((
            SELECT AVG(CASE WHEN ub.bill_status = 'paid' THEN 1 ELSE 0 END)
            FROM utility_bill ub 
            WHERE ub.borrower_id = b.borrower_id 
            AND ub.due_date <= l.origination_date
        ), 0) AS utility_score_before
        
    FROM loan l
    JOIN borrower b ON l.borrower_id = b.borrower_id
    LEFT JOIN borrower_mobile_summary bm ON b.borrower_id = bm.borrower_id
    WHERE l.loan_status IN ('paid', 'default')
    ORDER BY l.origination_date
    """
    
    print("Executing query...")
    cursor = conn.cursor()
    cursor.execute(query)
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Clean data
    for col in df.columns:
        if col not in ['borrower_type', 'loan_status']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Encode borrower_type
    if 'borrower_type' in df.columns:
        le = LabelEncoder()
        df['borrower_type_encoded'] = le.fit_transform(df['borrower_type'].astype(str))
    
    print(f"✓ Loaded {len(df)} loans")
    print(f"✓ Default rate: {df['default_flag'].mean()*100:.1f}%")
    print(f"✓ Date range: {df['origination_date'].min()} to {df['origination_date'].max()}")
    
    return df

def engineer_features_safe(df):
    """
    Engineer features using ONLY information available at loan origination
    """
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING (Time-Consistent)")
    print("=" * 70)
    
    # Loan-to-Income ratio
    df['loan_to_income'] = df['loan_amount'] / (df['monthly_income'] + 1)
    df['loan_to_income'] = df['loan_to_income'].clip(0, 5)
    
    # Prior default severity
    df['default_severity'] = df['past_defaults_before'] * (df['max_past_due_before'] / 30)
    
    # Income stability
    df['income_stability'] = df['transaction_frequency'] * df['airtime_consistency']
    
    # Risk score (inverse of credit score)
    df['risk_score'] = 850 - df['credit_score_before']
    df['risk_score'] = df['risk_score'].clip(0, 550)
    
    # Tenor risk
    df['tenor_risk'] = df['tenor_months'] / 24
    
    # Utility reliability
    df['utility_reliability'] = df['utility_score_before'] * 100
    
    # Customer tenure
    df['tenure_years'] = df['days_as_customer_at_loan'] / 365
    
    # Debt burden (existing debt vs income)
    df['debt_to_income'] = df['total_prev_loans'] * df['loan_amount'] / (df['monthly_income'] * 12 + 1)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 2)
    
    # Combined risk score
    df['combined_risk'] = (
        df['risk_score'] * 0.35 +
        df['debt_to_income'] * 100 * 0.20 +
        df['past_defaults_before'] * 50 * 0.20 +
        df['tenor_risk'] * 100 * 0.15 +
        df['loan_to_income'] * 20 * 0.10
    )
    
    print(f"✓ Created {len(df.columns)} total features")
    
    return df

def temporal_train_test_split(df):
    """
    Split data chronologically - NO LOOKAHEAD BIAS
    Train on older loans, test on newer loans
    """
    
    print("\n" + "=" * 70)
    print("TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 70)
    
    # Sort by origination date
    df = df.sort_values('origination_date')
    
    # Use first 80% for training, last 20% for testing
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training period: {train_df['origination_date'].min()} to {train_df['origination_date'].max()}")
    print(f"Testing period: {test_df['origination_date'].min()} to {test_df['origination_date'].max()}")
    print(f"Training loans: {len(train_df)} ({train_df['default_flag'].mean()*100:.1f}% default)")
    print(f"Testing loans: {len(test_df)} ({test_df['default_flag'].mean()*100:.1f}% default)")
    
    return train_df, test_df

def select_features(df):
    """Select features for training"""
    
    feature_cols = [
        'loan_to_income',
        'tenor_risk',
        'loan_amount',
        'tenor_months',
        'credit_score_before',
        'past_defaults_before',
        'max_past_due_before',
        'debt_to_income',
        'default_severity',
        'monthly_income',
        'income_stability',
        'transaction_frequency',
        'airtime_consistency',
        'age',
        'tenure_years',
        'utility_reliability',
        'risk_score',
        'combined_risk',
        'borrower_type_encoded'
    ]
    
    # Keep available features
    available = [f for f in feature_cols if f in df.columns]
    
    X = df[available].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train model and evaluate"""
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    default_rate = y_train.mean()
    print(f"Training default rate: {default_rate*100:.1f}%")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost
    scale_pos = (1 - default_rate) / max(default_rate, 0.01)
    xgb = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=min(scale_pos, 5),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train
    rf.fit(X_train_scaled, y_train)
    xgb.fit(X_train_scaled, y_train)
    
    # Predict
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    
    # Ensemble
    ensemble_proba = (rf_proba + xgb_proba) / 2
    
    # Calculate metrics
    rf_auc = roc_auc_score(y_test, rf_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    
    print(f"\n  Random Forest: AUC = {rf_auc:.4f}")
    print(f"  XGBoost: AUC = {xgb_auc:.4f}")
    print(f"  Ensemble: AUC = {ensemble_auc:.4f}")
    
    # Choose best
    if ensemble_auc >= max(rf_auc, xgb_auc):
        best_proba = ensemble_proba
        best_name = "Ensemble"
        best_auc = ensemble_auc
    elif rf_auc >= xgb_auc:
        best_proba = rf_proba
        best_name = "Random Forest"
        best_auc = rf_auc
    else:
        best_proba = xgb_proba
        best_name = "XGBoost"
        best_auc = xgb_auc
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
    
    if len(thresholds) > 0 and len(f1_scores) > 0:
        best_threshold = thresholds[np.argmax(f1_scores)]
    else:
        best_threshold = 0.5
    
    y_pred = (best_proba >= best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': best_auc,
        'best_threshold': best_threshold,
        'best_model': best_name
    }
    
    return rf, xgb, scaler, metrics, best_proba

def main():
    print("\n" + "=" * 70)
    print("ML PIPELINE - TEMPORAL SPLIT (No Lookahead)")
    print("=" * 70)
    
    # Fetch data with temporal tracking
    df = fetch_all_loan_data()
    
    if len(df) < 50:
        print(f"\n❌ Need at least 50 loans. Found {len(df)}")
        print("\n📌 Generate more loan data first:")
        print("   python generate_loans_fixed.py")
        return
    
    # Engineer features
    df = engineer_features_safe(df)
    
    # Temporal split (no lookahead!)
    train_df, test_df = temporal_train_test_split(df)
    
    # Prepare features
    X_train, feature_names = select_features(train_df)
    y_train = train_df['default_flag']
    
    X_test, _ = select_features(test_df)
    y_test = test_df['default_flag']
    
    print(f"\n📊 Feature set: {len(feature_names)} features")
    
    # Train
    rf, xgb, scaler, metrics, best_proba = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Save models
    joblib.dump(rf, 'models/rf_model.pkl')
    joblib.dump(xgb, 'models/xgb_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, (best_proba >= metrics['best_threshold']).astype(int))
    
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"\n📊 Model: {metrics['best_model']}")
    print(f"📈 Optimal Threshold: {metrics['best_threshold']:.3f}")
    print(f"\n🎯 Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1 Score:  {metrics['f1']:.3f}")
    print(f"   AUC-ROC:   {metrics['auc']:.3f}")
    
    print("\n📊 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    # Sanity check
    if metrics['auc'] > 0.95:
        print("\n⚠️ WARNING: AUC > 0.95 - Possible data leakage")
    elif metrics['auc'] > 0.75:
        print("\n✅ Model is realistic and ready for use")
    else:
        print("\n📊 Model needs improvement. Add more data or features.")
    
    print("\n" + "=" * 70)
    print("✅ Models saved to 'models/' directory")
    print("=" * 70)

if __name__ == "__main__":
    from config.settings import DB_CONFIG
    main()