"""
Complete ML Pipeline for Credit Default Prediction
Includes: Data cleaning, preprocessing, feature engineering, model training, and evaluation
"""

import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, f1_score, accuracy_score,
    recall_score, precision_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

class DataPreprocessor:
    """Handles all data cleaning and preprocessing steps"""
    
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.outlier_bounds = {}
        
    def detect_outliers_iqr(self, df, column, multiplier=1.5):
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return lower_bound, upper_bound
    
    def cap_outliers(self, df, column, lower_bound, upper_bound):
        """Cap outliers instead of removing them"""
        df[column] = df[column].clip(lower_bound, upper_bound)
        return df
    
    def clean_data(self, df):
        """Initial data cleaning"""
        print("\n🔧 Step 1: Initial Data Cleaning")
        
        original_shape = df.shape
        print(f"  Original shape: {original_shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"  After removing duplicates: {df.shape}")
        
        # Remove columns that are all null
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            df = df.drop(columns=null_cols)
            print(f"  Dropped columns with all nulls: {null_cols}")
        
        # Remove identifier columns
        id_columns = ['loan_id', 'borrower_id', 'full_name']
        df = df.drop(columns=[c for c in id_columns if c in df.columns], errors='ignore')
        
        return df
    
    def handle_missing_values(self, df):
        """Intelligent missing value handling"""
        print("\n🔧 Step 2: Handling Missing Values")
        
        missing_before = df.isnull().sum().sum()
        print(f"  Missing values before: {missing_before}")
        
        # For each column, decide strategy based on data type and missing percentage
        for col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            
            if missing_pct == 0:
                continue
                
            print(f"    {col}: {missing_pct:.1f}% missing")
            
            if missing_pct > 70:
                # Drop columns with >70% missing
                df = df.drop(columns=[col])
                print(f"      → Dropped (>{70}% missing)")
                
            elif col in ['loan_purpose', 'borrower_type', 'credit_score_band', 'age_group']:
                # Categorical columns - use mode
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                print(f"      → Filled categorical with mode: '{mode_val}'")
                
            else:
                # Numeric columns - use median (robust to outliers)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"      → Filled numeric with median: {median_val:.2f}")
        
        missing_after = df.isnull().sum().sum()
        print(f"  Missing values after: {missing_after}")
        
        return df
    
    def handle_outliers(self, df):
        """Cap outliers instead of removing"""
        print("\n🔧 Step 3: Handling Outliers")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_count = 0
        for col in numeric_cols:
            lower, upper = self.detect_outliers_iqr(df, col)
            self.outlier_bounds[col] = (lower, upper)
            
            before_count = ((df[col] < lower) | (df[col] > upper)).sum()
            if before_count > 0:
                df = self.cap_outliers(df, col, lower, upper)
                outlier_count += before_count
                print(f"    {col}: capped {before_count} outliers (range: {lower:.2f} - {upper:.2f})")
        
        print(f"  Total outliers capped: {outlier_count}")
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        print("\n🔧 Step 4: Encoding Categorical Variables")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"    {col}: encoded to {len(le.classes_)} categories")
        
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale features using RobustScaler (robust to outliers)"""
        print("\n🔧 Step 5: Feature Scaling")
        
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"    Scaled {X_train.shape[1]} features")
        return X_train_scaled, X_test_scaled
    
    def preprocess(self, df, is_training=True, X_train=None):
        """Run full preprocessing pipeline"""
        
        # Step 1: Clean data
        df = self.clean_data(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Handle outliers
        df = self.handle_outliers(df)
        
        # Step 4: Encode categorical
        df = self.encode_categorical(df)
        
        # Separate features and target
        if 'default_flag' in df.columns:
            y = df['default_flag']
            X = df.drop(columns=['default_flag'])
        else:
            X = df
            y = None
        
        self.feature_names = X.columns.tolist()
        
        if is_training and y is not None:
            # Scale features
            X_scaled, _ = self.scale_features(X, X)
            return X_scaled, y
        else:
            return X, None

class CreditDefaultModel:
    """Complete ML model with preprocessing pipeline"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.best_threshold = 0.5
        self.training_metrics = {}
        
    def load_data(self, db):
        """Load and prepare raw data from database"""
        
        print("\n📊 Loading Data from Database")
        print("=" * 50)
        
        query = """
        SELECT 
            -- Loan features
            l.loan_id,
            l.original_amount,
            l.loan_status,
            l.days_past_due,
            l.interest_rate,
            l.tenor_months,
            
            -- Borrower features
            b.borrower_id,
            b.borrower_type,
            TIMESTAMPDIFF(YEAR, b.date_of_birth, CURDATE()) AS age,
            DATEDIFF(CURDATE(), b.registration_date) AS days_as_customer,
            
            -- Credit history
            COALESCE(bc.num_active_loans, 0) AS active_loans,
            COALESCE(bc.total_outstanding_balance, 0) AS total_debt,
            COALESCE(bc.num_loans_past_due, 0) AS loans_past_due,
            COALESCE(bc.max_days_past_due, 0) AS max_days_past_due,
            COALESCE(bc.num_past_defaults, 0) AS past_defaults,
            COALESCE(bc.months_since_last_default, 999) AS months_since_default,
            
            -- Mobile money (alternative data)
            COALESCE(bm.avg_monthly_volume, 0) AS monthly_income,
            COALESCE(bm.avg_transaction_frequency, 0) AS transaction_frequency,
            COALESCE(bm.avg_airtime_topup_consistency, 0) AS airtime_consistency,
            
            -- Credit score
            COALESCE(cs.credit_score, 600) AS credit_score,
            
            -- Behavioral
            COALESCE(bf.night_applications, 0) AS night_applications,
            COALESCE(bf.total_applications_last_30_days, 0) AS recent_applications,
            
            -- Utility
            COALESCE((
                SELECT AVG(CASE WHEN bill_status = 'paid' THEN 1 ELSE 0 END)
                FROM utility_bill ub WHERE ub.borrower_id = b.borrower_id
            ), 0) AS utility_score,
            
            -- Loan purpose
            COALESCE(la.loan_purpose, 'Unknown') AS loan_purpose,
            
            -- Target
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
        
        conn = pymysql.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Basic data quality report
        print(f"\n📊 Data Quality Report:")
        print(f"  Default rate: {df['default_flag'].mean()*100:.1f}%")
        print(f"  Unique borrowers: {df['borrower_id'].nunique()}")
        print(f"  Date range: {df['origination_date'].min()} to {df['origination_date'].max()}")
        
        return df
    
    def engineer_features(self, df):
        """Create advanced derived features"""
        
        print("\n🔧 Step 6: Feature Engineering")
        print("=" * 50)
        
        # Debt-to-Income Ratio
        df['dti_ratio'] = df['total_debt'] / (df['monthly_income'] * 12 + 1)
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
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                                  labels=['Young', 'Early Career', 'Mid Career', 'Senior'])
        
        # Credit score bands
        df['credit_band'] = pd.cut(df['credit_score'], 
                                    bins=[0, 500, 600, 700, 850],
                                    labels=['Very Poor', 'Poor', 'Fair', 'Good'])
        
        print(f"✓ Created {len([c for c in df.columns if c not in ['default_flag']])} features")
        
        return df
    
    def select_features(self, X, y):
        """Select most important features based on correlation and variance"""
        
        print("\n🔧 Step 7: Feature Selection")
        print("=" * 50)
        
        # Remove low variance features
        variances = X.var()
        low_var_cols = variances[variances < 0.01].index.tolist()
        if low_var_cols:
            X = X.drop(columns=low_var_cols)
            print(f"  Removed {len(low_var_cols)} low-variance features")
        
        # Remove high correlation features
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if high_corr_cols:
            X = X.drop(columns=high_corr_cols)
            print(f"  Removed {len(high_corr_cols)} high-correlation features")
        
        print(f"  Final feature count: {X.shape[1]}")
        
        return X
    
    def balance_data(self, X_train, y_train):
        """Handle class imbalance using SMOTE"""
        
        print("\n🔧 Step 8: Handling Class Imbalance")
        print("=" * 50)
        
        original_ratio = y_train.mean()
        print(f"  Original default ratio: {original_ratio*100:.1f}%")
        
        # Use SMOTE with Tomek links for better boundary definition
        smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.4)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        
        new_ratio = y_resampled.mean()
        print(f"  Resampled default ratio: {new_ratio*100:.1f}%")
        print(f"  New training size: {len(X_resampled)} (was {len(X_train)})")
        
        return X_resampled, y_resampled
    
    def train(self, db):
        """Complete training pipeline"""
        
        print("\n" + "=" * 70)
        print("STARTING ML MODEL TRAINING PIPELINE")
        print("=" * 70)
        
        # 1. Load data
        df_raw = self.load_data(db)
        
        if len(df_raw) < 200:
            return False, f"Insufficient data: {len(df_raw)} records (need 200+)"
        
        # 2. Engineer features
        df = self.engineer_features(df_raw)
        
        # 3. Separate features and target
        y = df['default_flag']
        X = df.drop(columns=['default_flag', 'loan_id', 'borrower_id', 'origination_date', 
                              'loan_status', 'full_name'], errors='ignore')
        
        # 4. Preprocess
        X_processed, _ = self.preprocessor.preprocess(X, is_training=True, X_train=None)
        
        # 5. Feature selection
        X_selected = self.select_features(X_processed, y)
        
        # 6. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 7. Balance training data
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)
        
        # 8. Train models
        print("\n🔧 Step 9: Model Training")
        print("=" * 50)
        
        # Individual models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=25,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_train_balanced) - y_train_balanced.sum()) / y_train_balanced.sum(),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Train and evaluate
        models = {
            'Random Forest': rf,
            'XGBoost': xgb,
            'Gradient Boosting': gb
        }
        
        best_model = None
        best_auc = 0
        best_name = None
        
        for name, model in models.items():
            model.fit(X_train_balanced, y_train_balanced)
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            
            print(f"  {name}: AUC = {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_name = name
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
            voting='soft',
            weights=[2, 3, 2]  # XGBoost gets higher weight
        )
        ensemble.fit(X_train_balanced, y_train_balanced)
        ensemble_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
        print(f"  Ensemble (Voting): AUC = {ensemble_auc:.4f}")
        
        if ensemble_auc > best_auc:
            best_model = ensemble
            best_name = "Ensemble (Voting)"
            best_auc = ensemble_auc
        
        # 9. Find optimal threshold
        y_proba = best_model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
        self.best_threshold = thresholds[np.argmax(f1_scores)]
        
        # 10. Final evaluation with optimal threshold
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        self.model = best_model
        self.training_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': best_auc,
            'best_threshold': self.best_threshold,
            'best_model': best_name,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'default_rate': y_train.mean()
        }
        
        # 11. Save model and preprocessor
        joblib.dump(best_model, 'models/credit_default_model.pkl')
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        joblib.dump(self.best_threshold, 'models/best_threshold.pkl')
        
        # 12. Print final report
        self.print_final_report()
        
        return True, self.training_metrics
    
    def print_final_report(self):
        """Print comprehensive training report"""
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - FINAL REPORT")
        print("=" * 70)
        
        metrics = self.training_metrics
        
        print(f"\n📊 Model Information:")
        print(f"  Best Model: {metrics['best_model']}")
        print(f"  Optimal Threshold: {metrics['best_threshold']:.3f}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc']:.3f}")
        
        print(f"\n💰 Business Impact:")
        print(f"  Training samples: {metrics['training_samples']}")
        print(f"  Test samples: {metrics['test_samples']}")
        print(f"  Population default rate: {metrics['default_rate']*100:.1f}%")
        
        # Interpretability
        print(f"\n🎯 Interpretation:")
        if metrics['precision'] > 0.85:
            print(f"  ✅ High Precision: Few false approvals (good for lender risk)")
        elif metrics['precision'] > 0.70:
            print(f"  📊 Moderate Precision: Some false approvals, but manageable")
        else:
            print(f"  ⚠️ Low Precision: Many false approvals - consider raising threshold")
        
        if metrics['recall'] > 0.70:
            print(f"  ✅ High Recall: Catching most defaults (good for loss prevention)")
        elif metrics['recall'] > 0.50:
            print(f"  📊 Moderate Recall: Missing some defaults, but acceptable")
        else:
            print(f"  ⚠️ Low Recall: Missing many defaults - consider lowering threshold")
        
        print("\n" + "=" * 70)
        print("✅ Model saved to 'models/credit_default_model.pkl'")
        print("=" * 70)

def main():
    from utils.db_connection import get_db
    db = get_db()
    
    model = CreditDefaultModel()
    success, result = model.train(db)
    
    if success:
        print("\n🎉 Model training successful! You can now:")
        print("  1. Use the model for predictions on new borrowers")
        print("  2. Deploy the model via API")
        print("  3. Monitor model performance over time")
    else:
        print(f"\n❌ Training failed: {result}")

if __name__ == "__main__":
    from config.settings import DB_CONFIG
    main()