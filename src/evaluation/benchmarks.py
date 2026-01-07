"""
Comprehensive benchmarking suite
Tests all strategies across multiple data sizes and scenarios
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
from typing import Dict, List
from tqdm import tqdm
import json

from crypto.classical import ClassicalCrypto
from crypto.postquantum import PostQuantumCrypto
from migration.policy_engine import MigrationPolicyEngine


class ComprehensiveBenchmark:
    """Complete benchmarking system"""
    
    def __init__(self):
        self.test_sizes = {
            'tiny': 1024,           # 1 KB (single sensor reading)
            'small': 10 * 1024,     # 10 KB (batch of readings)
            'medium': 100 * 1024,   # 100 KB (hourly data)
            'large': 1024 * 1024,   # 1 MB (daily data)
            'xlarge': 10 * 1024 * 1024  # 10 MB (weekly data)
        }
        
        self.algorithms = {
            'classical': ['RSA-2048', 'RSA-3072', 'ECC-256'],
            'postquantum': ['Kyber512', 'Kyber768', 'Kyber1024']
        }
        
        self.results = []
        
    def generate_test_data(self, size: int) -> bytes:
        """Generate random test data of specified size"""
        return os.urandom(size)
    
    def benchmark_encryption(self, data: bytes, crypto_type: str, 
                            algorithm: str, iterations: int = 10) -> Dict:
        """
        Benchmark encryption performance
        Run multiple iterations for statistical reliability
        """
        metrics = {
            'crypto_type': crypto_type,
            'algorithm': algorithm,
            'data_size_bytes': len(data),
            'data_size_label': self._size_label(len(data)),
            'iterations': iterations
        }
        
        # Initialize crypto module
        if crypto_type == 'classical':
            crypto = ClassicalCrypto(algorithm)
        else:
            crypto = PostQuantumCrypto(algorithm)
        
        # Metrics storage
        keygen_times = []
        encrypt_times = []
        decrypt_times = []
        storage_overheads = []
        cpu_usages = []
        memory_usages = []
        
        process = psutil.Process()
        
        for _ in range(iterations):
            # Measure key generation
            start_mem = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent(interval=0.1)
            
            private_key, public_key, keygen_time = crypto.generate_keypair()
            keygen_times.append(keygen_time)
            
            # Measure encryption
            start_time = time.time()
            encrypted_blob, enc_metrics = crypto.encrypt_data(data, public_key)
            encrypt_time = time.time() - start_time
            encrypt_times.append(encrypt_time)
            storage_overheads.append(enc_metrics['storage_overhead_percent'])
            
            # Measure decryption
            start_time = time.time()
            if crypto_type == 'classical':
                decrypted_data, dec_metrics = crypto.decrypt_data(encrypted_blob, private_key)
            else:
                decrypted_data, dec_metrics = crypto.decrypt_data(encrypted_blob, private_key, public_key)
            decrypt_time = time.time() - start_time
            decrypt_times.append(decrypt_time)
            
            # Measure resource usage
            end_cpu = process.cpu_percent(interval=0.1)
            end_mem = process.memory_info().rss / 1024 / 1024
            
            cpu_usages.append(end_cpu - start_cpu)
            memory_usages.append(end_mem - start_mem)
            
            # Verify correctness
            assert decrypted_data == data, "Decryption failed!"
        
        # Calculate statistics
        metrics.update({
            'keygen_time_mean_ms': np.mean(keygen_times) * 1000,
            'keygen_time_std_ms': np.std(keygen_times) * 1000,
            'encrypt_time_mean_ms': np.mean(encrypt_times) * 1000,
            'encrypt_time_std_ms': np.std(encrypt_times) * 1000,
            'decrypt_time_mean_ms': np.mean(decrypt_times) * 1000,
            'decrypt_time_std_ms': np.std(decrypt_times) * 1000,
            'total_time_mean_ms': (np.mean(keygen_times) + np.mean(encrypt_times) + 
                                   np.mean(decrypt_times)) * 1000,
            'storage_overhead_mean_percent': np.mean(storage_overheads),
            'storage_overhead_std_percent': np.std(storage_overheads),
            'cpu_usage_mean_percent': np.mean(cpu_usages),
            'memory_usage_mean_mb': np.mean(memory_usages),
            'throughput_mbps': (len(data) / 1024 / 1024) / np.mean(encrypt_times)
        })
        
        return metrics
    
    def _size_label(self, size_bytes: int) -> str:
        """Get human-readable size label"""
        for label, size in self.test_sizes.items():
            if size_bytes == size:
                return label
        return f"{size_bytes / 1024:.1f}KB"
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("="*70)
        print("🚀 COMPREHENSIVE CRYPTOGRAPHIC BENCHMARK")
        print("="*70)
        
        total_tests = (len(self.test_sizes) * 
                      (len(self.algorithms['classical']) + 
                       len(self.algorithms['postquantum'])))
        
        print(f"\nRunning {total_tests} benchmark combinations...")
        print(f"Test sizes: {list(self.test_sizes.keys())}")
        print(f"Classical algorithms: {self.algorithms['classical']}")
        print(f"Post-Quantum algorithms: {self.algorithms['postquantum']}")
        
        pbar = tqdm(total=total_tests, desc="Benchmarking")
        
        # Test all combinations
        for size_label, size_bytes in self.test_sizes.items():
            test_data = self.generate_test_data(size_bytes)
            
            # Classical algorithms
            for algo in self.algorithms['classical']:
                try:
                    metrics = self.benchmark_encryption(test_data, 'classical', algo)
                    self.results.append(metrics)
                except Exception as e:
                    print(f"\n⚠️  Failed: {algo} on {size_label}: {e}")
                pbar.update(1)
            
            # Post-quantum algorithms
            for algo in self.algorithms['postquantum']:
                try:
                    metrics = self.benchmark_encryption(test_data, 'postquantum', algo)
                    self.results.append(metrics)
                except Exception as e:
                    print(f"\n⚠️  Failed: {algo} on {size_label}: {e}")
                pbar.update(1)
        
        pbar.close()
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        print("\n✅ Benchmark complete!")
        return self.results_df
    
    def save_results(self, filepath: str = 'results/metrics/full_benchmark.csv'):
        """Save benchmark results"""
        self.results_df.to_csv(filepath, index=False)
        print(f"✅ Saved results to {filepath}")
        
        # Also save as JSON
        json_path = filepath.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved JSON to {json_path}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("📊 BENCHMARK SUMMARY")
        print("="*70)
        
        # Group by algorithm and size
        summary = self.results_df.groupby(['crypto_type', 'algorithm', 'data_size_label']).agg({
            'encrypt_time_mean_ms': 'mean',
            'storage_overhead_mean_percent': 'mean',
            'throughput_mbps': 'mean'
        }).round(2)
        
        print("\n🔐 Classical Cryptography:")
        classical = summary.loc['classical']
        print(classical.to_string())
        
        print("\n🔮 Post-Quantum Cryptography:")
        pq = summary.loc['postquantum']
        print(pq.to_string())
        
        # Overall comparison
        print("\n📈 Classical vs Post-Quantum (Average across all sizes):")
        comparison = self.results_df.groupby('crypto_type').agg({
            'encrypt_time_mean_ms': 'mean',
            'storage_overhead_mean_percent': 'mean',
            'throughput_mbps': 'mean'
        }).round(2)
        print(comparison.to_string())


class MLGuidedVsBaseline:
    """Compare ML-guided migration vs baseline strategies"""
    
    def __init__(self):
        self.policy_engine = MigrationPolicyEngine()
        
    def compare_strategies(self, test_df: pd.DataFrame) -> Dict:
        """
        Compare three approaches:
        1. ML-guided selective migration
        2. Blanket post-quantum (migrate everything)
        3. Do nothing (keep classical)
        """
        print("\n" + "="*70)
        print("🧠 ML-GUIDED vs BASELINE COMPARISON")
        print("="*70)
        
        # Get ML recommendations
        ml_decisions = self.policy_engine.decide_strategy(test_df)
        
        # Calculate metrics for each approach
        results = {}
        
        # 1. ML-Guided
        ml_strategies = ml_decisions['migration_strategy'].value_counts()
        ml_pq_percent = (ml_strategies.get('postquantum', 0) / len(ml_decisions)) * 100
        ml_hybrid_percent = (ml_strategies.get('hybrid', 0) / len(ml_decisions)) * 100
        ml_classical_percent = (ml_strategies.get('classical', 0) / len(ml_decisions)) * 100
        
        # Estimate overhead (weighted average)
        ml_avg_overhead = (
            ml_pq_percent * 62.0 +      # PQ overhead
            ml_hybrid_percent * 40.0 +  # Hybrid (estimated)
            ml_classical_percent * 15.8 # Classical overhead
        ) / 100
        
        results['ml_guided'] = {
            'strategy': 'ML-Guided Selective',
            'pq_percent': ml_pq_percent,
            'hybrid_percent': ml_hybrid_percent,
            'classical_percent': ml_classical_percent,
            'avg_storage_overhead': ml_avg_overhead,
            'security_score': self._calculate_security_score(ml_decisions)
        }
        
        # 2. Blanket PQ
        results['blanket_pq'] = {
            'strategy': 'Blanket Post-Quantum',
            'pq_percent': 100.0,
            'hybrid_percent': 0.0,
            'classical_percent': 0.0,
            'avg_storage_overhead': 62.0,
            'security_score': 100.0  # Maximum security
        }
        
        # 3. Do Nothing
        results['do_nothing'] = {
            'strategy': 'Do Nothing (Classical)',
            'pq_percent': 0.0,
            'hybrid_percent': 0.0,
            'classical_percent': 100.0,
            'avg_storage_overhead': 15.8,
            'security_score': self._calculate_security_score_baseline(test_df)
        }
        
        # Print comparison
        comparison_df = pd.DataFrame(results).T
        print("\n📊 Strategy Comparison:")
        print(comparison_df.to_string())
        
        # Calculate efficiency gain
        overhead_reduction = ((results['blanket_pq']['avg_storage_overhead'] - 
                              results['ml_guided']['avg_storage_overhead']) / 
                             results['blanket_pq']['avg_storage_overhead'] * 100)
        
        security_retained = (results['ml_guided']['security_score'] / 
                           results['blanket_pq']['security_score'] * 100)
        
        print(f"\n💡 Key Findings:")
        print(f"   • ML reduces overhead by {overhead_reduction:.1f}% vs blanket PQ")
        print(f"   • ML retains {security_retained:.1f}% of maximum security")
        print(f"   • ML protects {ml_pq_percent:.1f}% of data with full PQ")
        
        return results
    
    def _calculate_security_score(self, decisions_df: pd.DataFrame) -> float:
        """
        Calculate security score based on correct risk-strategy matching
        """
        # Perfect match scoring
        correct_matches = 0
        total = len(decisions_df)
        
        for _, row in decisions_df.iterrows():
            risk = row['predicted_risk']
            strategy = row['migration_strategy']
            
            if risk == 'critical' and strategy == 'postquantum':
                correct_matches += 1
            elif risk == 'high' and strategy == 'postquantum':
                correct_matches += 1
            elif risk == 'medium' and strategy == 'hybrid':
                correct_matches += 1
            elif risk == 'low' and strategy == 'classical':
                correct_matches += 1
        
        return (correct_matches / total) * 100
    
    def _calculate_security_score_baseline(self, test_df: pd.DataFrame) -> float:
        """
        Security score for do-nothing approach
        Based on how many high/critical risk items are left unprotected
        """
        high_risk_count = len(test_df[test_df['risk_label'].isin(['high', 'critical'])])
        total = len(test_df)
        
        # Lower score because high-risk data isn't protected
        vulnerable_percent = (high_risk_count / total) * 100
        return 100 - vulnerable_percent  # Inverse of vulnerability


def run_all_benchmarks():
    """Execute complete benchmark suite"""
    
    # 1. Comprehensive crypto benchmarks
    print("\n" + "="*70)
    print("PHASE 7: COMPREHENSIVE BENCHMARKING")
    print("="*70)
    
    benchmark = ComprehensiveBenchmark()
    results_df = benchmark.run_full_benchmark()
    benchmark.save_results()
    benchmark.print_summary()
    
    # 2. ML-guided comparison
    print("\n\nLoading test dataset for ML comparison...")
    test_df = pd.read_csv('data/synthetic/iot_metadata_test.csv')
    
    ml_comparison = MLGuidedVsBaseline()
    comparison_results = ml_comparison.compare_strategies(test_df)
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv('results/metrics/ml_vs_baseline.csv')
    print("\n✅ Saved ML comparison to results/metrics/ml_vs_baseline.csv")
    
    return results_df, comparison_results


if __name__ == "__main__":
    run_all_benchmarks()