"""
Stacked Ensemble ML Model for XDSData Ghana
Combines multiple models for optimal precision and recall
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class StackedMLModel:
    """
    Stacked Ensemble Model for Credit Default Prediction
    Optimizes for both precision AND recall
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.best_threshold = 0.5
        self.model_performance = {}
        self.model_path = 'models/stacked_model.pkl'
        self.scaler_path = 'models/stacked_scaler.pkl'
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    def load_data_from_db(self, db, sample_limit=100000):
        """Load and prepare data from database"""
        
        query = f"""
        SELECT 
            -- Core credit features
            COALESCE(cs.credit_score, 0) AS credit_score,
            COALESCE(bc.num_past_defaults, 0) AS past_defaults,
            COALESCE(bc.max_days_past_due, 0) AS max_days_past_due,
            COALESCE(bc.num_active_loans, 0) AS active_loans,
            COALESCE(bc.total_outstanding_balance, 0) AS total_debt,
            COALESCE(bc.months_since_last_default, 999) AS months_since_default,
            COALESCE(bc.num_loans_past_due, 0) AS loans_past_due,
            
            -- Alternative data features
            COALESCE(bm.avg_monthly_volume, 0) AS monthly_income,
            COALESCE(bm.avg_transaction_frequency, 0) AS transaction_frequency,
            COALESCE(bm.avg_airtime_topup_consistency, 0) AS airtime_consistency,
            
            -- Behavioral features
            COALESCE(bf.night_applications, 0) AS night_applications,
            COALESCE(bf.total_applications_last_30_days, 0) AS recent_applications,
            COALESCE(bf.avg_loan_to_income_ratio, 0) AS loan_to_income_ratio,
            
            -- Utility features
            COALESCE((
                SELECT AVG(CASE WHEN bill_status = 'paid' THEN 1 ELSE 0 END)
                FROM utility_bill ub
                WHERE ub.borrower_id = b.borrower_id
            ), 0) AS utility_payment_score,
            
            -- Loan features
            la.loan_purpose,
            la.requested_amount,
            la.requested_tenor,
            
            -- Target variable
            CASE WHEN l.loan_status = 'default' THEN 1 ELSE 0 END AS default_flag
            
        FROM borrower b
        INNER JOIN loan l ON b.borrower_id = l.borrower_id
        LEFT JOIN borrower_credit_summary bc ON b.borrower_id = bc.borrower_id
        LEFT JOIN borrower_mobile_summary bm ON b.borrower_id = bm.borrower_id
        LEFT JOIN credit_score cs ON b.borrower_id = cs.borrower_id
        LEFT JOIN borrower_features_all bf ON b.borrower_id = bf.borrower_id
        LEFT JOIN loan_application la ON b.borrower_id = la.borrower_id
        WHERE l.loan_status IN ('paid', 'default')
        LIMIT {sample_limit}
        """
        
        result = db.execute_query(query)
        
        if result and len(result) > 0:
            df = pd.DataFrame(result)
            return df
        return pd.DataFrame()
    
    def engineer_features(self, df):
        """Create engineered features for better prediction"""
        
        # Calculate DTI ratio
        df['dti_ratio'] = df.apply(
            lambda x: (x['total_debt'] / (x['monthly_income'] * 12)) 
            if x['monthly_income'] > 0 else 1.0, axis=1
        )
        df['dti_ratio'] = df['dti_ratio'].clip(0, 2)
        
        # Monthly debt service
        df['monthly_debt_service'] = df['total_debt'] / 12
        
        # Available payment capacity
        df['available_payment'] = (df['monthly_income'] * 0.30) - df['monthly_debt_service']
        df['available_payment'] = df['available_payment'].clip(0, None)
        
        # Credit score bands
        df['credit_score_band'] = pd.cut(df['credit_score'], 
            bins=[0, 500, 600, 700, 850], 
            labels=['Very Poor', 'Poor', 'Fair', 'Good'])
        
        # Past default severity
        df['default_severity'] = df['past_defaults'] * df['max_days_past_due'] / 30
        
        # Income stability proxy
        df['income_stability'] = df['transaction_frequency'] * df['airtime_consistency']
        
        # Risk score (inverse of credit score)
        df['risk_score'] = 850 - df['credit_score']
        
        # Loan-to-income ratio
        df['requested_lti'] = df['requested_amount'] / df['monthly_income'].replace(0, 1)
        df['requested_lti'] = df['requested_lti'].clip(0, 5)
        
        # Interaction features
        df['debt_credit_interaction'] = df['total_debt'] * (850 - df['credit_score'])
        df['income_default_interaction'] = df['monthly_income'] * (1 + df['past_defaults'])
        
        # Recent distress indicator
        df['recent_distress'] = ((df['months_since_default'] < 12) & (df['past_defaults'] > 0)).astype(int)
        
        # Night application risk
        df['night_risk'] = df['night_applications'] * 0.1
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Define feature columns
        feature_columns = [
            'credit_score',
            'past_defaults', 
            'max_days_past_due',
            'active_loans',
            'total_debt',
            'monthly_income',
            'months_since_default',
            'transaction_frequency',
            'night_applications',
            'dti_ratio',
            'monthly_debt_service',
            'available_payment',
            'default_severity',
            'income_stability',
            'risk_score',
            'requested_lti',
            'debt_credit_interaction',
            'recent_distress',
            'night_risk',
            'utility_payment_score',
            'airtime_consistency',
            'loans_past_due'
        ]
        
        # Keep only available columns
        available_features = [f for f in feature_columns if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], 0)
        
        return X
    
    def create_stacked_model(self):
        """Create stacked ensemble model"""
        
        # Base models (Level 1)
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
        
        # Meta model (Level 2)
        meta_model = LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        # Create stacked model
        stacked_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacked_model
    
    def find_optimal_threshold(self, model, X_val, y_val):
        """Find optimal threshold to balance precision and recall"""
        
        y_proba = model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        self.best_threshold = best_threshold
        
        return {
            'threshold': best_threshold,
            'precision': best_precision,
            'recall': best_recall,
            'f1': best_f1,
            'threshold_results': results
        }
    
    def train(self, db, sample_limit=100000):
        """Train the stacked ensemble model"""
        
        try:
            # Load data
            df = self.load_data_from_db(db, sample_limit)
            
            if df.empty or len(df) < 500:
                return False, f"Insufficient data: {len(df)} records (need 500+)"
            
            # Prepare features
            X = self.prepare_features(df)
            y = df['default_flag'].values
            
            # Handle class imbalance with SMOTE-Tomek
            smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.4)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train stacked model
            self.model = self.create_stacked_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Find optimal threshold
            threshold_result = self.find_optimal_threshold(self.model, X_test_scaled, y_test)
            
            # Make predictions with optimal threshold
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_proba >= self.best_threshold).astype(int)
            
            # Calculate metrics
            self.model_performance = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba),
                'avg_precision': average_precision_score(y_test, y_proba),
                'optimal_threshold': self.best_threshold,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.is_trained = True
            
            # Save model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.best_threshold, 'models/best_threshold.pkl')
            
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            return True, {
                'performance': self.model_performance,
                'feature_importance': feature_importance,
                'threshold_result': threshold_result
            }
            
        except Exception as e:
            return False, str(e)
    
    def get_feature_importance(self):
        """Extract feature importance from models"""
        
        importance_dict = {}
        
        # Get importance from base models
        if hasattr(self.model, 'estimators_'):
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    for feat, imp in zip(self.feature_names, estimator.feature_importances_):
                        if feat not in importance_dict:
                            importance_dict[feat] = []
                        importance_dict[feat].append(imp)
        
        # Average importance across models
        avg_importance = {
            feat: np.mean(imps) for feat, imps in importance_dict.items()
        }
        
        # Sort by importance
        sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:15])  # Top 15 features
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.best_threshold = joblib.load('models/best_threshold.pkl')
                self.is_trained = True
                return True
        except:
            pass
        return False
    
    def predict(self, borrower_data):
        """Predict default probability and decision"""
        
        if not self.is_trained:
            return None, None, None
        
        try:
            # Prepare features
            X = pd.DataFrame([borrower_data])
            X = self.prepare_features(X)
            
            # Scale
            X_scaled = self.scaler.transform(X)
            
            # Get probability
            default_prob = self.model.predict_proba(X_scaled)[0][1]
            
            # Apply threshold
            approval_prob = 1 - default_prob
            decision = "APPROVE" if approval_prob >= self.best_threshold else "DECLINE"
            
            # Risk level based on probability
            if default_prob < 0.15:
                risk_level = "Very Low"
            elif default_prob < 0.30:
                risk_level = "Low"
            elif default_prob < 0.50:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return default_prob, decision, risk_level
            
        except Exception as e:
            return None, None, None
    
    def predict_batch(self, borrowers_df):
        """Predict for multiple borrowers"""
        
        if not self.is_trained:
            return None
        
        X = self.prepare_features(borrowers_df)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        decisions = ["APPROVE" if (1-p) >= self.best_threshold else "DECLINE" for p in probs]
        
        return probs, decisions

# Global instance
stacked_model = StackedMLModel()