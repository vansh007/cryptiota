"""
Synthetic IoT Metadata Generator
Generates realistic encrypted IoT data records with cryptographic risk labels
"""

import pandas as pd
import numpy as np
from faker import Faker
import json
from datetime import datetime, timedelta
from typing import List, Dict
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class IoTDataGenerator:
    """Generates synthetic IoT device metadata with encryption characteristics"""
    
    def __init__(self):
        # Define IoT device types with realistic characteristics
        self.device_types = {
            'health_monitor': {
                'data_types': ['heart_rate', 'blood_pressure', 'spo2', 'temperature'],
                'sampling_rate_range': (1, 10),  # Hz
                'data_size_range': (32, 128),  # KB per reading
                'confidentiality': 'critical',
                'retention_years': (10, 30),
                'regulatory': ['HIPAA', 'GDPR']
            },
            'smart_home_sensor': {
                'data_types': ['temperature', 'humidity', 'motion', 'door_status'],
                'sampling_rate_range': (0.1, 1),
                'data_size_range': (8, 32),
                'confidentiality': 'low',
                'retention_years': (1, 5),
                'regulatory': ['GDPR']
            },
            'industrial_monitor': {
                'data_types': ['pressure', 'vibration', 'flow_rate', 'temperature'],
                'sampling_rate_range': (10, 100),
                'data_size_range': (64, 256),
                'confidentiality': 'high',
                'retention_years': (5, 15),
                'regulatory': ['ISO27001', 'NIST']
            },
            'location_tracker': {
                'data_types': ['gps_coordinates', 'altitude', 'speed'],
                'sampling_rate_range': (0.1, 1),
                'data_size_range': (16, 64),
                'confidentiality': 'high',
                'retention_years': (3, 10),
                'regulatory': ['GDPR', 'CCPA']
            },
            'environmental_sensor': {
                'data_types': ['air_quality', 'co2', 'particulate_matter', 'noise'],
                'sampling_rate_range': (0.01, 0.1),
                'data_size_range': (16, 64),
                'confidentiality': 'low',
                'retention_years': (2, 7),
                'regulatory': []
            }
        }
        
        # Encryption algorithms with realistic adoption rates
        self.encryption_algorithms = {
            'RSA-2048': 0.35,  # Still widely deployed
            'RSA-3072': 0.15,
            'RSA-4096': 0.05,
            'ECC-256': 0.25,
            'ECC-384': 0.15,
            'AES-256': 0.05   # Symmetric only (already PQ-safe)
        }
        
        # Key rotation patterns (in days)
        self.key_rotation_patterns = [30, 90, 180, 365, 730, 1825]  # Never = 1825
    
    
    def _calculate_risk_label(self, record: Dict) -> str:
        """
        Ground truth with realistic uncertainty
        """
        algo = record['encryption_algorithm']
        retention = record['expected_retention_years']
        confidentiality = record['confidentiality_level']
        data_age = record['data_age_years']
    
        quantum_risk_years = {
            'RSA-2048': 5,
            'RSA-3072': 8,
            'RSA-4096': 12,
            'ECC-256': 6,
            'ECC-384': 10,
            'AES-256': 999
        }
    
        risk_horizon = quantum_risk_years.get(algo, 10)
        years_vulnerable = retention - risk_horizon
    
        # BASE SCORE CALCULATION (not deterministic)
        base_score = 0
    
        # Algorithm risk (30% weight)
        if algo in ['RSA-2048', 'ECC-256']:
            base_score += 30
        elif algo in ['RSA-3072']:
            base_score += 20
        elif algo in ['RSA-4096', 'ECC-384']:
            base_score += 10
    
        # Retention period (40% weight)
        if retention > 15:
            base_score += 40
        elif retention > 8:
            base_score += 30
        elif retention > 3:
            base_score += 15
        else:
            base_score += 5
    
        # Confidentiality (20% weight)
        if confidentiality == 'critical':
            base_score += 20
        elif confidentiality == 'high':
            base_score += 15
        elif confidentiality == 'medium':
            base_score += 7
    
        # Compliance (10% weight)
        if record['regulatory_requirement'] != 'None':
            base_score += 10
    
        # ADD REALISTIC NOISE (this makes ML necessary)
        import random
        noise = random.gauss(0, 5)  # Gaussian noise
        final_score = base_score + noise
    
        # FUZZY BOUNDARIES (overlap between classes)
        if final_score >= 75:
            return 'critical'
        elif final_score >= 55:
            return 'high'
        elif final_score >= 35:
            return 'medium'
        else:
            return 'low'
    
  
    def generate_records(self, n_records: int = 10000, seed: int = 42) -> pd.DataFrame:
        """Generate with configurable seed"""
        np.random.seed(seed)
        random.seed(seed)
        records = []
        
        for i in range(n_records):
            # Select device type
            device_type = np.random.choice(list(self.device_types.keys()))
            device_config = self.device_types[device_type]
            
            # Generate characteristics
            data_type = np.random.choice(device_config['data_types'])
            sampling_rate = np.random.uniform(*device_config['sampling_rate_range'])
            data_size = np.random.uniform(*device_config['data_size_range'])
            
            # Encryption details
            encryption_algo = np.random.choice(
                list(self.encryption_algorithms.keys()),
                p=list(self.encryption_algorithms.values())
            )
            
            key_size = int(encryption_algo.split('-')[1]) if '-' in encryption_algo else 256
            
            key_rotation = np.random.choice(self.key_rotation_patterns)
            
            # Temporal characteristics
            data_age = np.random.uniform(0.1, 8)  # Years since creation
            retention_min, retention_max = device_config['retention_years']
            expected_retention = np.random.uniform(retention_min, retention_max)
            
            # Confidentiality
            confidentiality = device_config['confidentiality']
            
            # Regulatory
            regulatory = ', '.join(device_config['regulatory']) if device_config['regulatory'] else 'None'
            
            # Access patterns
            access_frequency = np.random.choice(['hourly', 'daily', 'weekly', 'monthly', 'rarely'])
            
            # Build record
            record = {
                'device_id': f"{device_type}_{i:05d}",
                'device_type': device_type,
                'data_type': data_type,
                'data_size_kb': round(data_size, 2),
                'sampling_rate_hz': round(sampling_rate, 3),
                'encryption_algorithm': encryption_algo,
                'key_size_bits': key_size,
                'key_rotation_days': key_rotation,
                'data_age_years': round(data_age, 2),
                'expected_retention_years': round(expected_retention, 1),
                'confidentiality_level': confidentiality,
                'regulatory_requirement': regulatory,
                'access_frequency': access_frequency
            }
            
            # Calculate risk label (ground truth)
            record['risk_label'] = self._calculate_risk_label(record)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add some noise/complexity for ML to learn
        # This makes it non-trivial - ML should beat simple rules
        df['compliance_score'] = np.random.uniform(0.7, 1.0, len(df))
        df['device_uptime_days'] = np.random.randint(30, 3650, len(df))
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV"""
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(df)} records to {filepath}")
        
        # Print dataset statistics
        print("\n📊 Dataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"\nRisk distribution:")
        print(df['risk_label'].value_counts())
        print(f"\nDevice types:")
        print(df['device_type'].value_counts())
        print(f"\nEncryption algorithms:")
        print(df['encryption_algorithm'].value_counts())


def main():
    """Generate and save synthetic dataset"""
    generator = IoTDataGenerator()
     # Train: seed=42
     # Generate training dataset
    print("🔄 Generating training dataset (10,000 records)...")
    train_df = generator.generate_records(n_records=10000, seed=42)
    generator.save_dataset(train_df, 'data/synthetic/iot_metadata_train.csv')
    
    # Test: seed=123 (DIFFERENT)
    # Generate test dataset
    print("\n🔄 Generating test dataset (2,000 records)...")
    test_df = generator.generate_records(n_records=2000, seed=123)
    generator.save_dataset(test_df, 'data/synthetic/iot_metadata_test.csv')
    
    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()