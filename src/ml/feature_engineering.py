"""
Feature engineering for quantum risk prediction
Transforms raw metadata into ML-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple
import joblib


class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Create ML features from raw metadata
        
        Features we'll create:
        1. Numerical features (scaled)
        2. Categorical features (encoded)
        3. Derived features (risk indicators)
        """
        
        df_features = df.copy()
        
        # === DERIVED FEATURES (Risk Indicators) ===
        
        # 1. Quantum vulnerability score (algorithm-based)
        quantum_risk_map = {
            'RSA-2048': 1.0,   # Highest risk
            'ECC-256': 0.9,
            'RSA-3072': 0.7,
            'ECC-384': 0.6,
            'RSA-4096': 0.4,
            'AES-256': 0.0     # Already quantum-safe
        }
        df_features['algo_quantum_risk'] = df_features['encryption_algorithm'].map(quantum_risk_map)
        
        # 2. Time-to-quantum-break estimate
        # Estimated years until quantum computers can break this algorithm
        break_time_map = {
            'RSA-2048': 5,
            'ECC-256': 6,
            'RSA-3072': 8,
            'ECC-384': 10,
            'RSA-4096': 12,
            'AES-256': 999
        }
        df_features['years_until_break'] = df_features['encryption_algorithm'].map(break_time_map)
        
        # 3. Exposure window (how long data will be vulnerable)
        df_features['exposure_years'] = (
            df_features['expected_retention_years'] - 
            df_features['years_until_break']
        )
        
        # 4. Key rotation health
        df_features['key_rotation_health'] = np.where(
            df_features['key_rotation_days'] < 365, 1.0,
            np.where(df_features['key_rotation_days'] < 730, 0.5, 0.0)
        )
        
        # 5. Confidentiality score
        confidentiality_map = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
        df_features['confidentiality_score'] = df_features['confidentiality_level'].map(confidentiality_map)
        
        # 6. Has regulatory requirements
        df_features['has_regulation'] = (df_features['regulatory_requirement'] != 'None').astype(int)
        
        # 7. Data criticality (size * retention * confidentiality)
        df_features['data_criticality'] = (
            np.log1p(df_features['data_size_kb']) * 
            df_features['expected_retention_years'] * 
            df_features['confidentiality_score']
        )
        
        # 8. Access risk (frequently accessed = more exposure)
        access_risk_map = {
            'hourly': 1.0,
            'daily': 0.8,
            'weekly': 0.5,
            'monthly': 0.3,
            'rarely': 0.1
        }
        df_features['access_risk'] = df_features['access_frequency'].map(access_risk_map)
        
        # === ENCODE CATEGORICAL FEATURES ===
        
        categorical_cols = [
            'device_type',
            'data_type',
            'encryption_algorithm',
            'confidentiality_level',
            'access_frequency'
        ]
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_features[col])
            else:
                df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])
        
        # === SELECT FINAL FEATURES ===

        # IMPORTANT:
        # We intentionally EXCLUDE the following from ML to avoid data leakage:
        # - algo_quantum_risk
        # - years_until_break
        # - exposure_years
        #
        # These are still computed above for:
        # - explainability
        # - policy engine logic
        # - analysis / reporting

        numerical_features = [
            'data_size_kb',
            'sampling_rate_hz',
            'key_size_bits',
            'key_rotation_days',
            'data_age_years',
            'expected_retention_years',
            'compliance_score',
            'device_uptime_days',

            # Derived but INDIRECT features (safe)
            'key_rotation_health',
            'confidentiality_score',
            'has_regulation',
            'data_criticality',
            'access_risk'
        ]

        encoded_features = [
            'device_type_encoded',
            'data_type_encoded',
            'encryption_algorithm_encoded',
            'confidentiality_level_encoded',
            'access_frequency_encoded'
        ]

        self.feature_names = numerical_features + encoded_features

        X = df_features[self.feature_names].copy()

        
        # === SCALE NUMERICAL FEATURES ===
        
        if fit:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        else:
            X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        return X
    
    def save(self, filepath: str):
        """Save feature engineering artifacts"""
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(artifacts, filepath)
        print(f"✅ Saved feature engineering artifacts to {filepath}")
    
    def load(self, filepath: str):
        """Load feature engineering artifacts"""
        artifacts = joblib.load(filepath)
        self.label_encoders = artifacts['label_encoders']
        self.scaler = artifacts['scaler']
        self.feature_names = artifacts['feature_names']
        print(f"✅ Loaded feature engineering artifacts from {filepath}")


def test_feature_engineering():
    """Test feature engineering"""
    print("🧪 Testing Feature Engineering...")
    
    # Load data
    df = pd.read_csv('data/synthetic/iot_metadata_train.csv')
    print(f"Loaded {len(df)} records")
    
    # Engineer features
    engineer = FeatureEngineer()
    X = engineer.engineer_features(df, fit=True)
    
    print(f"\nEngineered {X.shape[1]} features:")
    print(X.head())
    
    print(f"\nFeature names:")
    for i, name in enumerate(engineer.feature_names, 1):
        print(f"{i}. {name}")
    
    # Save
    engineer.save('models/feature_engineer.joblib')
    
    print("\n✅ Feature engineering test passed!")


if __name__ == "__main__":
    test_feature_engineering()