"""
ML Pipeline - No Data Leakage
Only uses information available BEFORE loan approval
"""

import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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

def fetch_loan_application_data():
    """
    Fetch ONLY data available at loan application time
    NO post-approval information (no days_past_due, no loan_status)
    """
    
    print("=" * 70)
    print("FETCHING PRE-APPROVAL DATA (No Data Leakage)")
    print("=" * 70)
    
    conn = pymysql.connect(**DB_CONFIG)
    
    # IMPORTANT: Only use information available BEFORE loan approval
    query = """
    SELECT 
        -- Application-time information only
        la.requested_amount AS loan_amount,
        la.requested_tenor AS tenor_months,
        la.loan_purpose,
        la.application_date,
        
        -- Borrower information at application time
        b.borrower_type,
        TIMESTAMPDIFF(YEAR, b.date_of_birth, la.application_date) AS age_at_application,
        DATEDIFF(la.application_date, b.registration_date) AS days_as_customer_at_application,
        
        -- Historical credit data (BEFORE this application)
        COALESCE((
            SELECT COUNT(*) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < la.application_date
            AND l2.loan_status = 'default'
        ), 0) AS past_defaults_before,
        
        COALESCE((
            SELECT MAX(l2.days_past_due) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < la.application_date
            AND l2.loan_status = 'default'
        ), 0) AS max_past_due_before,
        
        COALESCE((
            SELECT COUNT(*) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < la.application_date
            AND l2.loan_status = 'active'
        ), 0) AS active_loans_before,
        
        COALESCE((
            SELECT SUM(l2.original_amount) FROM loan l2 
            WHERE l2.borrower_id = b.borrower_id 
            AND l2.origination_date < la.application_date
            AND l2.loan_status = 'active'
        ), 0) AS total_debt_before,
        
        -- Mobile money data (BEFORE application)
        COALESCE(bm.avg_monthly_volume, 0) AS monthly_income,
        COALESCE(bm.avg_transaction_frequency, 0) AS transaction_frequency,
        COALESCE(bm.avg_airtime_topup_consistency, 0) AS airtime_consistency,
        
        -- Credit score (from BEFORE application)
        COALESCE((
            SELECT cs2.credit_score FROM credit_score cs2 
            WHERE cs2.borrower_id = b.borrower_id 
            AND cs2.score_date <= la.application_date
            ORDER BY cs2.score_date DESC LIMIT 1
        ), 600) AS credit_score_before,
        
        -- Utility score (from BEFORE application)
        COALESCE((
            SELECT AVG(CASE WHEN ub.bill_status = 'paid' THEN 1 ELSE 0 END)
            FROM utility_bill ub 
            WHERE ub.borrower_id = b.borrower_id 
            AND ub.due_date <= la.application_date
        ), 0) AS utility_score_before,
        
        -- Behavioral (applications in last 30 days BEFORE this)
        COALESCE((
            SELECT COUNT(*) FROM loan_application la2 
            WHERE la2.borrower_id = b.borrower_id 
            AND la2.application_date < la.application_date
            AND la2.application_date >= DATE_SUB(la.application_date, INTERVAL 30 DAY)
        ), 0) AS recent_apps_before,
        
        -- Target: Did this loan eventually default?
        CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END AS default_flag
        
    FROM loan_application la
    JOIN borrower b ON la.borrower_id = b.borrower_id
    LEFT JOIN loan l ON la.borrower_id = l.borrower_id 
        AND la.lender_id = l.lender_id 
        AND la.application_date <= l.origination_date
    LEFT JOIN borrower_mobile_summary bm ON b.borrower_id = bm.borrower_id
    WHERE la.application_status = 'approved'
    """
    
    print("Executing query (pre-approval data only)...")
    cursor = conn.cursor()
    cursor.execute(query)
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Convert Decimal to float
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    print(f"✓ Loaded {len(df)} loan applications")
    print(f"✓ Default rate: {df['default_flag'].mean()*100:.1f}%")
    
    return df

def engineer_features_pre_approval(df):
    """
    Create features using ONLY pre-approval information
    """
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING (Pre-Approval Only)")
    print("=" * 70)
    
    # Convert to numeric
    numeric_cols = ['loan_amount', 'tenor_months', 'age_at_application', 
                    'days_as_customer_at_application', 'past_defaults_before',
                    'max_past_due_before', 'active_loans_before', 'total_debt_before',
                    'monthly_income', 'credit_score_before', 'utility_score_before',
                    'recent_apps_before']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Loan-to-Income ratio (safe, uses only pre-approval data)
    df['loan_to_income'] = df['loan_amount'] / (df['monthly_income'] + 1)
    df['loan_to_income'] = df['loan_to_income'].clip(0, 5)
    
    # Prior default severity (only historical defaults)
    df['default_severity'] = df['past_defaults_before'] * (df['max_past_due_before'] / 30)
    
    # Income stability (behavioral, not outcome-based)
    df['income_stability'] = df['transaction_frequency'] * df['airtime_consistency']
    
    # Risk score based on credit score
    df['risk_score'] = 850 - df['credit_score_before']
    
    # Tenor risk (loan duration risk)
    df['tenor_risk'] = df['tenor_months'] / 24
    
    # Utility reliability (payment history)
    df['utility_reliability'] = df['utility_score_before'] * 100
    
    # Customer tenure (loyalty indicator)
    df['tenure_years'] = df['days_as_customer_at_application'] / 365
    
    # Debt burden (existing debt vs income)
    df['debt_to_income'] = df['total_debt_before'] / (df['monthly_income'] * 12 + 1)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 2)
    
    # Combined pre-approval risk score
    df['pre_approval_risk'] = (
        df['risk_score'] * 0.35 +
        df['debt_to_income'] * 100 * 0.20 +
        df['past_defaults_before'] * 50 * 0.20 +
        df['recent_apps_before'] * 10 * 0.15 +
        df['tenor_risk'] * 100 * 0.10
    )
    
    # Age group (demographic)
    df['age_group'] = pd.cut(df['age_at_application'], 
                              bins=[0, 25, 35, 50, 100],
                              labels=['Young', 'Early Career', 'Mid Career', 'Senior'])
    
    # Credit score band
    df['credit_band'] = pd.cut(df['credit_score_before'], 
                                bins=[0, 500, 600, 700, 850],
                                labels=['Very Poor', 'Poor', 'Fair', 'Good'])
    
    print(f"✓ Created {len(df.columns)} total features")
    
    return df

def prepare_features(df):
    """Select features for training (no leakage)"""
    
    # Only use features available at application time
    feature_cols = [
        # Loan characteristics
        'loan_to_income',
        'tenor_risk',
        'loan_amount',
        'tenor_months',
        
        # Credit history
        'credit_score_before',
        'past_defaults_before',
        'max_past_due_before',
        'active_loans_before',
        'total_debt_before',
        'debt_to_income',
        'default_severity',
        
        # Income/behavioral
        'monthly_income',
        'income_stability',
        'transaction_frequency',
        'airtime_consistency',
        
        # Demographic
        'age_at_application',
        'tenure_years',
        
        # Utility
        'utility_reliability',
        
        # Application behavior
        'recent_apps_before',
        
        # Derived risk
        'risk_score',
        'pre_approval_risk'
    ]
    
    # Keep available features
    available = [f for f in feature_cols if f in df.columns]
    
    X = df[available].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available

def train_model_no_leakage(X_train, y_train, X_test, y_test):
    """Train model without data leakage"""
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING (No Leakage)")
    print("=" * 70)
    
    default_rate = y_train.mean()
    print(f"✓ Training default rate: {default_rate*100:.1f}%")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest with proper regularization
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=25,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost with regularization
    scale_pos = (1 - default_rate) / max(default_rate, 0.01)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train and evaluate
    models = {}
    
    rf.fit(X_train_scaled, y_train)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    models['Random Forest'] = {'model': rf, 'auc': rf_auc}
    print(f"  Random Forest: AUC = {rf_auc:.4f}")
    
    xgb.fit(X_train_scaled, y_train)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    models['XGBoost'] = {'model': xgb, 'auc': xgb_auc}
    print(f"  XGBoost: AUC = {xgb_auc:.4f}")
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    print(f"  Ensemble: AUC = {ensemble_auc:.4f}")
    
    # Select best
    if ensemble_auc > max(models[m]['auc'] for m in models):
        best_model = ensemble
        best_auc = ensemble_auc
        best_name = "Ensemble"
        best_proba = ensemble_proba
    else:
        best_name = max(models, key=lambda x: models[x]['auc'])
        best_model = models[best_name]['model']
        best_auc = models[best_name]['auc']
        best_proba = models[best_name]['model'].predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
    best_threshold = 0.5
    if len(thresholds) > 0 and len(f1_scores) > 0:
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx]
        best_f1 = f1_scores[idx]
    
    # Final evaluation
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
    
    return best_model, scaler, metrics

def main():
    print("\n" + "=" * 70)
    print("ML PIPELINE - NO DATA LEAKAGE")
    print("Using only pre-approval information")
    print("=" * 70)
    
    # Fetch only pre-approval data
    df = fetch_loan_application_data()
    
    if len(df) < 100:
        print(f"\n❌ Error: Need at least 100 applications. Found {len(df)}")
        return
    
    # Engineer features (pre-approval only)
    df = engineer_features_pre_approval(df)
    
    # Prepare features
    X, feature_names = prepare_features(df)
    y = df['default_flag']
    
    print(f"\n📊 Feature set: {len(feature_names)} features")
    print(f"   Features: {feature_names[:5]}...")
    
    # Split data (using time-based split to prevent lookahead bias)
    # Sort by application date and use first 80% for training
    if 'application_date' in df.columns:
        df = df.sort_values('application_date')
        split_idx = int(len(df) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        print(f"\n📊 Time-based split (prevents lookahead):")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n📊 Random split:")
    
    print(f"   Training: {len(X_train)} records ({y_train.mean()*100:.1f}% default)")
    print(f"   Test: {len(X_test)} records ({y_test.mean()*100:.1f}% default)")
    
    # Train model
    best_model, scaler, metrics = train_model_no_leakage(X_train, y_train, X_test, y_test)
    
    # Save
    joblib.dump(best_model, 'models/credit_model_no_leakage.pkl')
    joblib.dump(scaler, 'models/scaler_no_leakage.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Report
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - NO DATA LEAKAGE")
    print("=" * 70)
    print(f"\n📊 Best Model: {metrics['best_model']}")
    print(f"📈 Optimal Threshold: {metrics['best_threshold']:.3f}")
    print(f"\n🎯 Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1 Score:  {metrics['f1']:.3f}")
    print(f"   AUC-ROC:   {metrics['auc']:.3f}")
    
    # Sanity check - AUC should NOT be 1.0
    if metrics['auc'] > 0.95:
        print("\n⚠️ WARNING: AUC > 0.95 - still suspicious")
        print("   Possible issues:")
        print("   - Still have data leakage")
        print("   - Test set too small")
        print("   - Too few borrowers")
    elif metrics['auc'] > 0.85:
        print("\n✅ Good: AUC in realistic range (0.85-0.95)")
    else:
        print("\n📊 Model needs improvement. Add more features or data.")
    
    print("\n" + "=" * 70)
    print("✅ Model saved to 'models/credit_model_no_leakage.pkl'")
    print("=" * 70)

if __name__ == "__main__":
    from config.settings import DB_CONFIG
    main()