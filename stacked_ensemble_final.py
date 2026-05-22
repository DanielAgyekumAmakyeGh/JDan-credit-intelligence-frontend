"""
Stacked Ensemble Model for Credit Default Prediction
As specified in the research proposal:
- Level 1: XGBoost, Random Forest, Gradient Boosting, AdaBoost
- Level 2: Logistic Regression (Meta-model)
- SMOTE-Tomek for class imbalance
"""

import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

# ============================================================
# SECTION 1: DATA EXTRACTION
# ============================================================

def extract_data():
    """Extract borrower features and default flags from database"""
    
    print("=" * 70)
    print("STACKED ENSEMBLE - CREDIT DEFAULT PREDICTION")
    print("=" * 70)
    
    conn = pymysql.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        b.borrower_id,
        b.borrower_type,
        TIMESTAMPDIFF(YEAR, b.date_of_birth, CURDATE()) AS age,
        DATEDIFF(l.origination_date, b.registration_date) AS customer_tenure_days,
        
        -- Credit history features
        COALESCE(bc.num_active_loans, 0) AS active_loans,
        COALESCE(bc.total_outstanding_balance, 0) AS total_debt,
        COALESCE(bc.num_loans_past_due, 0) AS loans_past_due,
        COALESCE(bc.max_days_past_due, 0) AS max_days_past_due,
        COALESCE(bc.num_past_defaults, 0) AS past_defaults,
        COALESCE(bc.months_since_last_default, 999) AS months_since_default,
        
        -- Alternative data (Mobile Money)
        COALESCE(bm.avg_monthly_volume, 0) AS monthly_income,
        COALESCE(bm.avg_transaction_frequency, 0) AS transaction_frequency,
        COALESCE(bm.avg_airtime_topup_consistency, 0) AS airtime_consistency,
        
        -- Credit score
        COALESCE(cs.credit_score, 650) AS credit_score,
        
        -- Behavioral features
        COALESCE(bf.night_applications, 0) AS night_applications,
        COALESCE(bf.total_applications_last_30_days, 0) AS recent_applications,
        COALESCE(bf.avg_loan_to_income_ratio, 0) AS loan_to_income_ratio,
        
        -- Utility features
        COALESCE((
            SELECT AVG(CASE WHEN bill_status = 'paid' THEN 1 ELSE 0 END)
            FROM utility_bill ub WHERE ub.borrower_id = b.borrower_id
        ), 0) AS utility_score,
        
        -- Loan features
        l.original_amount,
        l.interest_rate,
        l.tenor_months,
        
        -- Target variable
        CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END AS default_flag
        
    FROM loan l
    JOIN borrower b ON l.borrower_id = b.borrower_id
    LEFT JOIN borrower_credit_summary bc ON b.borrower_id = bc.borrower_id
    LEFT JOIN borrower_mobile_summary bm ON b.borrower_id = bm.borrower_id
    LEFT JOIN credit_score cs ON b.borrower_id = cs.borrower_id
    LEFT JOIN borrower_features_all bf ON b.borrower_id = bf.borrower_id
    WHERE l.loan_status IN ('paid', 'default')
    """
    
    print("\nExtracting data from MySQL...")
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Clean data
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"Extracted {len(df)} loan records")
    print(f"Default rate: {df['default_flag'].mean()*100:.1f}%")
    
    return df

# ============================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """Create derived features for better prediction"""
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    # Debt-to-Income Ratio (DTI) - Industry standard (30% safe threshold)
    df['dti_ratio'] = df['total_debt'] / (df['monthly_income'] * 12 + 1)
    df['dti_ratio'] = df['dti_ratio'].clip(0, 2)
    
    # Monthly debt service
    df['monthly_debt'] = df['total_debt'] / 12
    df['available_income'] = (df['monthly_income'] * 0.30) - df['monthly_debt']
    df['available_income'] = df['available_income'].clip(0, None)
    
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
    
    # Night application risk (behavioral indicator)
    df['night_risk'] = df['night_applications'] * 0.1
    
    # Tenor risk (longer loans = higher risk)
    df['tenor_risk'] = df['tenor_months'] / 24
    
    # Interest rate risk
    df['interest_risk'] = df['interest_rate'] / 30
    
    # Amount risk
    df['amount_risk'] = df['original_amount'] / 50000
    df['amount_risk'] = df['amount_risk'].clip(0, 1)
    
    # Interaction features
    df['debt_credit_interaction'] = df['total_debt'] * df['risk_score']
    df['income_default_interaction'] = df['monthly_income'] * (1 + df['past_defaults'])
    
    # Customer tenure risk
    df['tenure_risk'] = np.exp(-df['customer_tenure_days'] / 365)
    
    # Combined risk score
    df['combined_risk'] = (
        df['risk_score'] * 0.30 +
        df['dti_ratio'] * 100 * 0.25 +
        df['past_defaults'] * 50 * 0.20 +
        df['default_severity'] * 0.15 +
        df['tenor_risk'] * 100 * 0.10
    )
    
    print(f"Created {len(df.columns)} total features")
    
    # Display feature correlations with target
    corr = df.corr(numeric_only=True)['default_flag'].abs().sort_values(ascending=False)
    print("\nTop 10 features correlated with default:")
    for feat, val in corr[1:11].items():
        print(f"   {feat}: {val:.4f}")
    
    return df

# ============================================================
# SECTION 3: STACKED ENSEMBLE MODEL
# ============================================================

def create_stacked_ensemble():
    """
    Create stacked ensemble as specified in research proposal:
    - Base models: XGBoost, Random Forest, Gradient Boosting, AdaBoost
    - Meta-model: Logistic Regression
    """
    
    # Level 1: Base Models
    base_models = [
        ('xgb', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )),
        ('ada', AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        ))
    ]
    
    # Level 2: Meta-model
    meta_model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    return base_models, meta_model

def train_stacked_ensemble(X_train, y_train, X_test, y_test):
    """
    Train stacked ensemble with cross-validation
    Uses SMOTE-Tomek for class imbalance
    """
    
    print("\n" + "=" * 70)
    print("STACKED ENSEMBLE TRAINING")
    print("=" * 70)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE-Tomek (as specified in report)
    print("\nHandling class imbalance with SMOTE-Tomek...")
    smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.4)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
    print(f"   Original training size: {len(X_train_scaled)}")
    print(f"   Resampled training size: {len(X_train_balanced)}")
    print(f"   New default rate: {y_train_balanced.mean()*100:.1f}%")
    
    # Create base models
    base_models, meta_model = create_stacked_ensemble()
    
    # Level 1: Train base models and collect predictions
    print("\nLevel 1: Training Base Models...")
    
    # For storing predictions
    train_meta_features = np.zeros((X_train_balanced.shape[0], len(base_models)))
    test_meta_features = np.zeros((X_test_scaled.shape[0], len(base_models)))
    
    trained_base_models = {}
    
    # Use cross-validation to prevent overfitting
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (name, model) in enumerate(base_models):
        print(f"\n   Training {name}...")
        
        # Train on full balanced dataset
        model.fit(X_train_balanced, y_train_balanced)
        trained_base_models[name] = model
        
        # Get predictions on test set
        test_pred = model.predict_proba(X_test_scaled)[:, 1]
        test_meta_features[:, i] = test_pred
        
        # Cross-validation predictions for training (out-of-fold)
        oof_preds = np.zeros(X_train_balanced.shape[0])
        for train_idx, val_idx in kf.split(X_train_balanced, y_train_balanced):
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_train_balanced[train_idx], y_train_balanced.iloc[train_idx])
            oof_preds[val_idx] = fold_model.predict_proba(X_train_balanced[val_idx])[:, 1]
        
        train_meta_features[:, i] = oof_preds
        
        # Evaluate this base model
        cv_auc = roc_auc_score(y_train_balanced, oof_preds)
        print(f"      CV AUC: {cv_auc:.4f}")
    
    # Level 2: Train meta-model
    print("\nLevel 2: Training Meta-Model (Logistic Regression)...")
    meta_model.fit(train_meta_features, y_train_balanced)
    
    # Get meta-model coefficients
    print("\n   Meta-model weights for each base model:")
    for i, (name, _) in enumerate(base_models):
        print(f"      {name}: {meta_model.coef_[0][i]:.4f}")
    
    # Final predictions
    final_proba = meta_model.predict_proba(test_meta_features)[:, 1]
    
    return trained_base_models, meta_model, scaler, final_proba

# ============================================================
# SECTION 4: MODEL EVALUATION
# ============================================================

def evaluate_model(y_test, y_proba, model_name="Stacked Ensemble"):
    """Comprehensive model evaluation"""
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    best_f1 = f1_scores[best_idx]
    
    # Predict at optimal threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'best_threshold': best_threshold,
        'best_f1': best_f1
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_pred

def print_report(metrics, cm, model_name="Stacked Ensemble"):
    """Print formatted evaluation report"""
    
    print("\n" + "=" * 70)
    print(f"{model_name} - PERFORMANCE REPORT")
    print("=" * 70)
    
    print(f"\nModel Configuration:")
    print(f"   Base Models: XGBoost, Random Forest, Gradient Boosting, AdaBoost")
    print(f"   Meta-Model: Logistic Regression")
    print(f"   Imbalance Handling: SMOTE-Tomek")
    
    print(f"\nPerformance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {metrics['auc']:.4f}")
    print(f"   Optimal Threshold: {metrics['best_threshold']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"   Actual Paid    -> Predicted Paid: {cm[0,0]:5d} | Predicted Default: {cm[0,1]:5d}")
    print(f"   Actual Default -> Predicted Paid: {cm[1,0]:5d} | Predicted Default: {cm[1,1]:5d}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if metrics['auc'] >= 0.85:
        print(f"   Excellent discrimination (AUC > 0.85)")
    elif metrics['auc'] >= 0.75:
        print(f"   Good discrimination (AUC > 0.75)")
    elif metrics['auc'] >= 0.65:
        print(f"   Acceptable discrimination (AUC > 0.65)")
    else:
        print(f"   Poor discrimination - model needs improvement")

# ============================================================
# SECTION 5: MAIN PIPELINE
# ============================================================

def main():
    """Execute complete stacked ensemble pipeline"""
    
    # Step 1: Extract data
    df = extract_data()
    
    if len(df) < 200:
        print(f"\nInsufficient data: {len(df)} records (need 200+)")
        return
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Prepare feature matrix
    feature_cols = [
        'active_loans', 'total_debt', 'loans_past_due', 'max_days_past_due',
        'past_defaults', 'months_since_default', 'monthly_income',
        'transaction_frequency', 'airtime_consistency', 'credit_score',
        'night_applications', 'recent_applications', 'utility_score',
        'original_amount', 'interest_rate', 'tenor_months', 'age',
        'dti_ratio', 'available_income', 'default_severity', 'income_stability',
        'risk_score', 'loan_to_income', 'recent_distress', 'night_risk',
        'tenor_risk', 'interest_risk', 'amount_risk', 'debt_credit_interaction',
        'income_default_interaction', 'tenure_risk', 'combined_risk'
    ]
    
    # Keep available features
    available = [f for f in feature_cols if f in df.columns]
    X = df[available]
    y = df['default_flag']
    
    print(f"\nFeature matrix: {X.shape[1]} features, {X.shape[0]} samples")
    
    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData Split:")
    print(f"   Training: {len(X_train)} samples ({y_train.mean()*100:.1f}% default)")
    print(f"   Test: {len(X_test)} samples ({y_test.mean()*100:.1f}% default)")
    
    # Step 5: Train stacked ensemble
    base_models, meta_model, scaler, final_proba = train_stacked_ensemble(
        X_train, y_train, X_test, y_test
    )
    
    # Step 6: Evaluate
    metrics, cm, _ = evaluate_model(y_test, final_proba, "Stacked Ensemble")
    print_report(metrics, cm, "Stacked Ensemble")
    
    # Step 7: Save model artifacts
    joblib.dump({
        'base_models': base_models,
        'meta_model': meta_model,
        'scaler': scaler,
        'feature_names': available,
        'best_threshold': metrics['best_threshold'],
        'performance': metrics,
        'confusion_matrix': cm.tolist()
    }, 'models/stacked_ensemble_final.pkl')
    
    print("\n" + "=" * 70)
    print("Stacked Ensemble saved to 'models/stacked_ensemble_final.pkl'")
    print("=" * 70)
    
    # Step 8: Compare with individual base models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in base_models.items():
        proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"   {name}: AUC = {auc:.4f}")
    
    print(f"\n   Stacked Ensemble: AUC = {metrics['auc']:.4f}")

if __name__ == "__main__":
    from config.settings import DB_CONFIG
    main()