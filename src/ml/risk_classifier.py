"""
ML Risk Classifier
Predicts quantum risk level: Low, Medium, High, Critical
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from ml.feature_engineering import FeatureEngineer


class RiskClassifier:
    """ML model for quantum risk classification"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.label_encoder = None
        self.risk_levels = ['low', 'medium', 'high', 'critical']
        
    def _get_model(self):
        """Initialize ML model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, train_df: pd.DataFrame) -> Dict:
        """Train the risk classifier"""
        print(f"🚀 Training {self.model_type} classifier...")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_df['risk_label'])
        
        # Engineer features
        X_train = self.feature_engineer.engineer_features(train_df, fit=True)
        
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # Initialize model
        self.model = self._get_model()
        
        # Cross-validation
        print("\n📊 Cross-validation (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full dataset
        print("\n🔄 Training on full dataset...")
        self.model.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"✅ Training accuracy: {train_accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_engineer.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📈 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        metrics = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'feature_importance': feature_importance
        }
        
        return metrics
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate model on test set"""
        print("\n🧪 Evaluating on test set...")
        
        # Prepare test data
        y_test = self.label_encoder.transform(test_df['risk_label'])
        X_test = self.feature_engineer.engineer_features(test_df, fit=False)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
        }
        
        # Per-class metrics
        for i, risk_level in enumerate(self.risk_levels):
            y_test_binary = (y_test == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            metrics[f'precision_{risk_level}'] = precision_score(y_test_binary, y_pred_binary, zero_division=0)
            metrics[f'recall_{risk_level}'] = recall_score(y_test_binary, y_pred_binary, zero_division=0)
            metrics[f'f1_{risk_level}'] = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        
        # Print results
        print(f"\n{'='*50}")
        print("📊 TEST SET PERFORMANCE")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall:    {metrics['recall_macro']:.4f}")
        print(f"F1-Score:  {metrics['f1_macro']:.4f}")
        
        print(f"\n{'='*50}")
        print("📊 PER-CLASS PERFORMANCE")
        print(f"{'='*50}")
        for risk_level in self.risk_levels:
            print(f"\n{risk_level.upper()}:")
            print(f"  Precision: {metrics[f'precision_{risk_level}']:.4f}")
            print(f"  Recall:    {metrics[f'recall_{risk_level}']:.4f}")
            print(f"  F1-Score:  {metrics[f'f1_{risk_level}']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        print(f"\n{'='*50}")
        print("📊 CONFUSION MATRIX")
        print(f"{'='*50}")
        cm_df = pd.DataFrame(cm, 
                            index=[f'True_{l}' for l in self.risk_levels],
                            columns=[f'Pred_{l}' for l in self.risk_levels])
        print(cm_df)
        
        # Detailed classification report
        print(f"\n{'='*50}")
        print("📊 DETAILED CLASSIFICATION REPORT")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=self.risk_levels))
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict risk levels"""
        X = self.feature_engineer.engineer_features(data, fit=False)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Convert back to labels
        risk_labels = self.label_encoder.inverse_transform(predictions)
        
        return risk_labels, probabilities
    
    def save(self, filepath: str):
        """Save model and artifacts"""
        artifacts = {
            'model': self.model,
            'model_type': self.model_type,
            'label_encoder': self.label_encoder,
            'risk_levels': self.risk_levels
        }
        joblib.dump(artifacts, filepath)
        
        # Save feature engineer separately
        fe_path = filepath.replace('.joblib', '_feature_engineer.joblib')
        self.feature_engineer.save(fe_path)
        
        print(f"✅ Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load model and artifacts"""
        artifacts = joblib.load(filepath)
        self.model = artifacts['model']
        self.model_type = artifacts['model_type']
        self.label_encoder = artifacts['label_encoder']
        self.risk_levels = artifacts['risk_levels']
        
        # Load feature engineer
        fe_path = filepath.replace('.joblib', '_feature_engineer.joblib')
        self.feature_engineer.load(fe_path)
        
        print(f"✅ Loaded model from {filepath}")


def train_and_evaluate():
    """Complete training and evaluation pipeline"""
    print("="*60)
    print("🚀 ML RISK CLASSIFIER TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv('data/synthetic/iot_metadata_train.csv')
    test_df = pd.read_csv('data/synthetic/iot_metadata_test.csv')
    print(f"Train: {len(train_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    
    # Train model
    classifier = RiskClassifier(model_type='random_forest')
    train_metrics = classifier.train(train_df)
    
    # Evaluate
    test_metrics = classifier.evaluate(test_df)
    
    # Save model
    classifier.save('models/risk_classifier.joblib')
    
    # Save metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    # Save to file
    import json
    with open('results/metrics/classifier_metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else 
               v.to_dict() if isinstance(v, pd.DataFrame) else v
            for k, v in all_metrics.items()
        }
        json.dump(metrics_serializable, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.2%}")
    
    return classifier, all_metrics


if __name__ == "__main__":
    classifier, metrics = train_and_evaluate()