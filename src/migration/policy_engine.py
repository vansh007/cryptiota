"""
Migration Policy Engine
Decides which cryptographic strategy to use based on ML risk predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time

from ml.risk_classifier import RiskClassifier
from crypto.classical import ClassicalCrypto
from crypto.postquantum import PostQuantumCrypto


class MigrationPolicyEngine:
    """Decides and executes migration strategy"""
    
    def __init__(self, classifier_path: str = 'models/risk_classifier.joblib'):
        # Load ML classifier
        self.classifier = RiskClassifier()
        self.classifier.load(classifier_path)
        
        # Define policy mappings
        self.risk_to_strategy = {
            'low': 'classical',
            'medium': 'hybrid',
            'high': 'postquantum',
            'critical': 'postquantum'
        }
        
        # Initialize crypto modules
        self.crypto_modules = {
            'classical': {},
            'postquantum': {},
            'hybrid': {}
        }
    
    def decide_strategy(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decide migration strategy for each record
        Returns dataframe with added 'migration_strategy' column
        """
        print("🧠 Running ML risk prediction...")
        
        # Predict risks
        risk_predictions, risk_probabilities = self.classifier.predict(metadata_df)
        
        # Map to strategies
        strategies = [self.risk_to_strategy[risk] for risk in risk_predictions]
        
        # Add to dataframe
        result_df = metadata_df.copy()
        result_df['predicted_risk'] = risk_predictions
        result_df['migration_strategy'] = strategies
        
        # Add confidence scores
        max_probs = np.max(risk_probabilities, axis=1)
        result_df['prediction_confidence'] = max_probs
        
        # Statistics
        print(f"\n📊 Migration Strategy Distribution:")
        print(result_df['migration_strategy'].value_counts())
        print(f"\nAverage Confidence: {max_probs.mean():.2%}")
        
        return result_df
    
    def execute_migration(self, data: bytes, strategy: str, 
                         algorithm: str = None) -> Tuple[Dict, Dict]:
        """
        Execute encryption with chosen strategy
        Returns: (encrypted_data, metrics)
        """
        
        if strategy == 'classical':
            if algorithm is None:
                algorithm = 'RSA-2048'
            
            if algorithm not in self.crypto_modules['classical']:
                self.crypto_modules['classical'][algorithm] = ClassicalCrypto(algorithm)
            
            crypto = self.crypto_modules['classical'][algorithm]
            private_key, public_key, keygen_time = crypto.generate_keypair()
            encrypted_blob, enc_metrics = crypto.encrypt_data(data, public_key)
            
            metrics = {
                'strategy': 'classical',
                'algorithm': algorithm,
                'keygen_time': keygen_time,
                **enc_metrics
            }
            
            return encrypted_blob, metrics
        
        elif strategy == 'postquantum':
            pq_algo = 'Kyber768'
            
            if pq_algo not in self.crypto_modules['postquantum']:
                self.crypto_modules['postquantum'][pq_algo] = PostQuantumCrypto(pq_algo)
            
            crypto = self.crypto_modules['postquantum'][pq_algo]
            private_key, public_key, keygen_time = crypto.generate_keypair()
            encrypted_blob, enc_metrics = crypto.encrypt_data(data, public_key)
            
            metrics = {
                'strategy': 'postquantum',
                'algorithm': pq_algo,
                'keygen_time': keygen_time,
                **enc_metrics
            }
            
            return encrypted_blob, metrics
        
        elif strategy == 'hybrid':
            # Hybrid: Classical + PQ
            # For simplicity, we'll use PQ with note that it's hybrid
            # In production, you'd combine both
            pq_algo = 'Kyber768'
            
            if pq_algo not in self.crypto_modules['postquantum']:
                self.crypto_modules['postquantum'][pq_algo] = PostQuantumCrypto(pq_algo)
            
            crypto = self.crypto_modules['postquantum'][pq_algo]
            private_key, public_key, keygen_time = crypto.generate_keypair()
            encrypted_blob, enc_metrics = crypto.encrypt_data(data, public_key)
            
            metrics = {
                'strategy': 'hybrid',
                'algorithm': f'ECDH+{pq_algo}',
                'keygen_time': keygen_time,
                **enc_metrics
            }
            
            return encrypted_blob, metrics
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def compare_strategies(self, test_data: bytes, size_label: str = "test") -> pd.DataFrame:
        """
        Compare all three strategies on same data
        Returns performance comparison dataframe
        """
        print(f"\n⚖️  Comparing strategies on {size_label} data ({len(test_data)} bytes)...")
        
        results = []
        
        for strategy in ['classical', 'postquantum', 'hybrid']:
            print(f"\nTesting {strategy}...")
            _, metrics = self.execute_migration(test_data, strategy)
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE COMPARISON - {size_label}")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        
        return comparison_df


def test_policy_engine():
    """Test migration policy engine"""
    print("="*60)
    print("🧪 TESTING MIGRATION POLICY ENGINE")
    print("="*60)
    
    # Load test data
    test_df = pd.read_csv('data/synthetic/iot_metadata_test.csv').head(100)
    
    # Initialize engine
    engine = MigrationPolicyEngine()
    
    # Decide strategies
    result_df = engine.decide_strategy(test_df)
    
    # Show sample decisions
    print(f"\n📋 Sample Migration Decisions:")
    print(result_df[['device_type', 'encryption_algorithm', 'expected_retention_years', 
                     'predicted_risk', 'migration_strategy', 'prediction_confidence']].head(10))
    
    # Compare strategies on sample data
    test_data = b"IoT sensor reading" * 100  # ~1.8KB
    comparison_df = engine.compare_strategies(test_data, "1.8KB")
    
    # Save comparison
    comparison_df.to_csv('results/metrics/strategy_comparison.csv', index=False)
    
    print("\n✅ Policy engine test complete!")


if __name__ == "__main__":
    test_policy_engine()