# Add this section to display model predictions using the stacked ensemble

def load_stacked_ensemble():
    """Load the trained stacked ensemble model"""
    import joblib
    import os
    
    model_path = 'models/stacked_ensemble_final.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def predict_with_ensemble(model_data, features):
    """Make prediction using stacked ensemble"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    base_models = model_data.get('base_models', {})
    meta_model = model_data.get('meta_model')
    scaler = model_data.get('scaler')
    best_threshold = model_data.get('best_threshold', 0.5)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get base model predictions
    base_predictions = []
    for name, model in base_models.items():
        proba = model.predict_proba(features_scaled)[:, 1]
        base_predictions.append(proba.reshape(-1, 1))
    
    # Stack predictions
    meta_features = np.hstack(base_predictions)
    
    # Final prediction
    final_proba = meta_model.predict_proba(meta_features)[:, 1]
    decision = "APPROVE" if final_proba[0] < best_threshold else "DECLINE"
    
    return final_proba[0], decision, best_threshold