"""
QuantumGuard AI - Flask Backend API
Production-ready REST API for IoT quantum risk assessment
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
import os

# Import your ML models
import sys
# Absolute project root (ml-pq-iot-system)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add src/ to Python path so ML code works
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


from ml.risk_classifier import RiskClassifier
from migration.policy_engine import MigrationPolicyEngine
from crypto.classical import ClassicalCrypto
from crypto.postquantum import PostQuantumCrypto

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
  # Enable CORS for frontend

# Initialize models (load once at startup)
print("Loading ML models...")
risk_classifier = RiskClassifier()
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_classifier.joblib")
risk_classifier.load(MODEL_PATH)
FEATURE_ENGINEER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_engineer.joblib")


policy_engine = MigrationPolicyEngine()

print("✅ Models loaded successfully!")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'QuantumGuard AI',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """
    Main analysis endpoint
    Accepts CSV file upload and returns risk assessment
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate required columns
        required_cols = [
            'device_type', 'encryption_algorithm', 
            'expected_retention_years', 'confidentiality_level'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {missing_cols}'
            }), 400
        
        # Run ML risk assessment
        predictions, probabilities = risk_classifier.predict(df)
        
        # Get migration decisions
        df_with_decisions = policy_engine.decide_strategy(df)
        
        # Calculate statistics
        total_records = len(df)
        risk_distribution = df_with_decisions['predicted_risk'].value_counts().to_dict()
        strategy_distribution = df_with_decisions['migration_strategy'].value_counts().to_dict()
        
        # Calculate savings
        pq_percent = (strategy_distribution.get('postquantum', 0) / total_records) * 100
        hybrid_percent = (strategy_distribution.get('hybrid', 0) / total_records) * 100
        classical_percent = (strategy_distribution.get('classical', 0) / total_records) * 100
        
        # Weighted average overhead
        ml_avg_overhead = (
            pq_percent * 62.0 +
            hybrid_percent * 40.0 +
            classical_percent * 15.8
        ) / 100
        
        blanket_pq_overhead = 62.0
        cost_saving = ((blanket_pq_overhead - ml_avg_overhead) / blanket_pq_overhead) * 100
        
        # Security score (simplified)
        high_critical_count = (
            risk_distribution.get('high', 0) + 
            risk_distribution.get('critical', 0)
        )
        security_retained = ((high_critical_count / total_records) * 100) if total_records > 0 else 0
        security_retained = min(security_retained * 1.5, 95)  # Cap at 95%
        
        # Sample predictions (top 10)
        sample_predictions = []
        for idx in range(min(10, len(df_with_decisions))):
            row = df_with_decisions.iloc[idx]
            sample_predictions.append({
                'device': f"{row['device_type']} #{row.get('device_id', idx)}",
                'risk': row['predicted_risk'],
                'current': row['encryption_algorithm'],
                'recommended': get_recommendation_text(row['migration_strategy']),
                'retention': f"{row['expected_retention_years']:.1f} years",
                'confidence': f"{row['prediction_confidence']:.1%}"
            })
        
        # Build response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_records': total_records,
                'risk_distribution': risk_distribution,
                'strategy_distribution': strategy_distribution,
                'savings': {
                    'storage_reduction_percent': round(cost_saving, 1),
                    'cost_saving_percent': round(cost_saving * 1.5, 1),  # Estimate
                    'security_retained_percent': round(security_retained, 1),
                    'ml_avg_overhead_percent': round(ml_avg_overhead, 1),
                    'blanket_pq_overhead_percent': round(blanket_pq_overhead, 1)
                }
            },
            'predictions': sample_predictions,
            'recommendations': {
                'postquantum': strategy_distribution.get('postquantum', 0),
                'hybrid': strategy_distribution.get('hybrid', 0),
                'classical': strategy_distribution.get('classical', 0)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/api/export/migration-plan', methods=['POST'])
def export_migration_plan():
    """
    Export migration decisions as CSV
    """
    try:
        data = request.json
        if not data or 'predictions' not in data:
            return jsonify({'error': 'No analysis data provided'}), 400

        df = pd.DataFrame(data['predictions'])

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='migration_plan.csv'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo', methods=['GET'])
def demo_data():
    """
    Return demo analysis results without file upload
    """
    # Simulated demo results
    response = {
        'success': True,
        'is_demo': True,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_records': 2500,
            'risk_distribution': {
                'critical': 325,
                'high': 875,
                'medium': 750,
                'low': 550
            },
            'strategy_distribution': {
                'postquantum': 1200,
                'hybrid': 750,
                'classical': 550
            },
            'savings': {
                'storage_reduction_percent': 35.2,
                'cost_saving_percent': 68.4,
                'security_retained_percent': 91.3,
                'ml_avg_overhead_percent': 40.2,
                'blanket_pq_overhead_percent': 62.0
            }
        },
        'predictions': [
            {
                'device': 'Health Monitor #A2301',
                'risk': 'critical',
                'current': 'RSA-2048',
                'recommended': 'Kyber768 (Post-Quantum)',
                'retention': '20.0 years',
                'confidence': '99.2%'
            },
            {
                'device': 'Smart Home Sensor #B7721',
                'risk': 'low',
                'current': 'ECC-256',
                'recommended': 'Keep Classical',
                'retention': '2.5 years',
                'confidence': '96.8%'
            },
            {
                'device': 'Industrial Monitor #C9912',
                'risk': 'high',
                'current': 'RSA-3072',
                'recommended': 'Kyber768 (Post-Quantum)',
                'retention': '12.3 years',
                'confidence': '88.5%'
            },
            {
                'device': 'Location Tracker #D4432',
                'risk': 'medium',
                'current': 'ECC-384',
                'recommended': 'Hybrid (ECDH+Kyber)',
                'retention': '5.7 years',
                'confidence': '92.1%'
            }
        ],
        'recommendations': {
            'postquantum': 1200,
            'hybrid': 750,
            'classical': 550
        }
    }
    
    return jsonify(response)


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """
    Run crypto performance benchmark
    """
    try:
        data = request.json
        algorithm = data.get('algorithm', 'RSA-2048')
        data_size = data.get('data_size', 1024)  # bytes
        
        # Generate test data
        test_data = os.urandom(data_size)
        
        # Benchmark
        if algorithm in ['RSA-2048', 'RSA-3072', 'ECC-256']:
            crypto = ClassicalCrypto(algorithm)
            crypto_type = 'classical'
        else:
            crypto = PostQuantumCrypto(algorithm)
            crypto_type = 'postquantum'
        
        # Run encryption
        import time
        start = time.time()
        private_key, public_key, keygen_time = crypto.generate_keypair()
        encrypted_blob, enc_metrics = crypto.encrypt_data(test_data, public_key)
        
        if crypto_type == 'classical':
            decrypted_data, dec_metrics = crypto.decrypt_data(encrypted_blob, private_key)
        else:
            decrypted_data, dec_metrics = crypto.decrypt_data(encrypted_blob, private_key, public_key)
        
        total_time = time.time() - start
        
        result = {
            'success': True,
            'algorithm': algorithm,
            'crypto_type': crypto_type,
            'data_size_bytes': data_size,
            'keygen_time_ms': round(keygen_time * 1000, 2),
            'encrypt_time_ms': round(enc_metrics['total_enc_time'] * 1000, 2),
            'decrypt_time_ms': round(dec_metrics['total_dec_time'] * 1000, 2),
            'total_time_ms': round(total_time * 1000, 2),
            'storage_overhead_percent': round(enc_metrics['storage_overhead_percent'], 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/export/report', methods=['POST'])
def export_report():
    """
    Download full analysis report as JSON file
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No report data'}), 400

        output = io.StringIO()
        json.dump(data, output, indent=2)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='application/json',
            as_attachment=True,
            download_name='quantumguard_report.json'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500



def get_recommendation_text(strategy):
    """Convert strategy code to human-readable text"""
    mapping = {
        'postquantum': 'Kyber768 (Post-Quantum)',
        'hybrid': 'Hybrid (ECDH+Kyber)',
        'classical': 'Keep Classical'
    }
    return mapping.get(strategy, strategy)


if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting QuantumGuard AI Backend API")
    print("="*60)
    print("\n📡 Endpoints:")
    print("  • GET  /api/health        - Health check")
    print("  • POST /api/analyze       - Upload CSV for analysis")
    print("  • GET  /api/demo          - Get demo results")
    print("  • POST /api/benchmark     - Run crypto benchmark")
    print("  • POST /api/export/report - Generate report")
    print("\n✅ Server ready!")
    print("="*60 + "\n")
    
    # app.run(debug=True, host='0.0.0.0', port=5001)
    if __name__ == "__main__":
    app.run()
