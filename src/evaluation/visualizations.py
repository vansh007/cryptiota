"""
Paper-ready visualizations
Generates all figures for your paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PaperVisualizations:
    """Generate all paper figures"""
    
    def __init__(self):
        self.output_dir = 'results/figures/'
        
    def plot_confusion_matrix(self, y_true, y_pred, labels, 
                              filename='confusion_matrix.png'):
        """Figure 1: Confusion Matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Risk Level')
        ax.set_ylabel('True Risk Level')
        ax.set_title('Risk Classification Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def plot_feature_importance(self, feature_names, importances, 
                               top_n=15, filename='feature_importance.png'):
        """Figure 2: Feature Importance"""
        # Sort features
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance for Risk Classification')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def plot_performance_comparison(self, benchmark_df, 
                                   filename='performance_comparison.png'):
        """Figure 3: Encryption Performance Comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prepare data
        classical = benchmark_df[benchmark_df['crypto_type'] == 'classical']
        pq = benchmark_df[benchmark_df['crypto_type'] == 'postquantum']
        
        size_order = ['tiny', 'small', 'medium', 'large', 'xlarge']
        
        # 1. Encryption Time
        ax = axes[0, 0]
        for algo in classical['algorithm'].unique():
            data = classical[classical['algorithm'] == algo]
            data = data.set_index('data_size_label').reindex(size_order)
            ax.plot(data.index, data['encrypt_time_mean_ms'], 
                   marker='o', label=f'{algo} (Classical)')
        
        for algo in pq['algorithm'].unique():
            data = pq[pq['algorithm'] == algo]
            data = data.set_index('data_size_label').reindex(size_order)
            ax.plot(data.index, data['encrypt_time_mean_ms'], 
                   marker='s', linestyle='--', label=f'{algo} (PQ)')
        
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Encryption Time (ms)')
        ax.set_title('Encryption Performance')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        
        # 2. Storage Overhead
        ax = axes[0, 1]
        classical_overhead = classical.groupby('data_size_label')['storage_overhead_mean_percent'].mean()
        pq_overhead = pq.groupby('data_size_label')['storage_overhead_mean_percent'].mean()
        
        x = np.arange(len(size_order))
        width = 0.35
        
        ax.bar(x - width/2, [classical_overhead.get(s, 0) for s in size_order], 
              width, label='Classical', color='steelblue')
        ax.bar(x + width/2, [pq_overhead.get(s, 0) for s in size_order], 
              width, label='Post-Quantum', color='coral')
        
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Storage Overhead (%)')
        ax.set_title('Storage Overhead Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(size_order)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Throughput
        ax = axes[1, 0]
        classical_throughput = classical.groupby('data_size_label')['throughput_mbps'].mean()
        pq_throughput = pq.groupby('data_size_label')['throughput_mbps'].mean()
        
        ax.bar(x - width/2, [classical_throughput.get(s, 0) for s in size_order], 
              width, label='Classical', color='steelblue')
        ax.bar(x + width/2, [pq_throughput.get(s, 0) for s in size_order], 
              width, label='Post-Quantum', color='coral')
        
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title('Encryption Throughput')
        ax.set_xticks(x)
        ax.set_xticklabels(size_order)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Key Generation Time
        ax = axes[1, 1]
        classical_keygen = classical.groupby('algorithm')['keygen_time_mean_ms'].mean()
        pq_keygen = pq.groupby('algorithm')['keygen_time_mean_ms'].mean()
        
        all_algos = list(classical_keygen.index) + list(pq_keygen.index)
        all_times = list(classical_keygen.values) + list(pq_keygen.values)
        colors = ['steelblue'] * len(classical_keygen) + ['coral'] * len(pq_keygen)
        
        ax.barh(range(len(all_algos)), all_times, color=colors)
        ax.set_yticks(range(len(all_algos)))
        ax.set_yticklabels(all_algos)
        ax.set_xlabel('Key Generation Time (ms)')
        ax.set_title('Key Generation Performance')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def plot_ml_vs_baseline(self, comparison_df, 
                           filename='ml_vs_baseline.png'):
        """Figure 4: ML-Guided vs Baseline Strategies"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        strategies = comparison_df.index.tolist()
        
        # 1. Storage Overhead
        ax = axes[0]
        overheads = comparison_df['avg_storage_overhead'].values
        colors = ['green', 'red', 'orange']
        
        bars = ax.bar(strategies, overheads, color=colors, alpha=0.7)
        ax.set_ylabel('Average Storage Overhead (%)')
        ax.set_title('Storage Overhead by Strategy')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Security Score
        ax = axes[1]
        security_scores = comparison_df['security_score'].values
        
        bars = ax.bar(strategies, security_scores, color=colors, alpha=0.7)
        ax.set_ylabel('Security Score (%)')
        ax.set_title('Security Level by Strategy')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def plot_risk_distribution(self, test_df, predictions, 
                              filename='risk_distribution.png'):
        """Figure 5: Risk Distribution - True vs Predicted"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        risk_levels = ['low', 'medium', 'high', 'critical']
        
        # True distribution
        ax = axes[0]
        true_counts = test_df['risk_label'].value_counts().reindex(risk_levels)
        ax.pie(true_counts, labels=risk_levels, autopct='%1.1f%%',
              colors=sns.color_palette("RdYlGn_r", 4))
        ax.set_title('True Risk Distribution')
        
        # Predicted distribution
        ax = axes[1]
        pred_series = pd.Series(predictions)
        pred_counts = pred_series.value_counts().reindex(risk_levels)
        ax.pie(pred_counts, labels=risk_levels, autopct='%1.1f%%',
              colors=sns.color_palette("RdYlGn_r", 4))
        ax.set_title('Predicted Risk Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def plot_cost_benefit_analysis(self, comparison_df,
                                   filename='cost_benefit.png'):
        """Figure 6: Cost-Benefit Trade-off"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = comparison_df['avg_storage_overhead'].values
        y = comparison_df['security_score'].values
        labels = comparison_df.index.tolist()
        colors = ['green', 'red', 'orange']
        
        for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
            ax.scatter(xi, yi, s=300, c=colors[i], alpha=0.6, edgecolors='black')
            ax.annotate(label, (xi, yi), fontsize=10, ha='center', va='center')
        
        ax.set_xlabel('Storage Overhead (%)')
        ax.set_ylabel('Security Score (%)')
        ax.set_title('Cost-Benefit Trade-off Analysis')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 70])
        ax.set_ylim([50, 105])
        
        # Add Pareto frontier line
        sorted_points = sorted(zip(x, y))
        ax.plot([p[0] for p in sorted_points], [p[1] for p in sorted_points], 
               'k--', alpha=0.3, label='Pareto Frontier')
        
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()


def generate_all_figures():
    """Generate all paper figures"""
    print("\n" + "="*70)
    print("PHASE 8: GENERATING PAPER VISUALIZATIONS")
    print("="*70)
    
    viz = PaperVisualizations()
    
    # Load data
    print("\n📂 Loading data...")
    
    # Load ML metrics
    with open('results/metrics/classifier_metrics.json', 'r') as f:
        ml_metrics = json.load(f)
    
    # Load test data
    test_df = pd.read_csv('data/synthetic/iot_metadata_test.csv')
    
    # Load benchmark results
    benchmark_df = pd.read_csv('results/metrics/full_benchmark.csv')
    
    # Load comparison
    comparison_df = pd.read_csv('results/metrics/ml_vs_baseline.csv', index_col=0)
    
    # Load trained model for predictions
    from ml.risk_classifier import RiskClassifier
    classifier = RiskClassifier()
    classifier.load('models/risk_classifier.joblib')
    predictions, _ = classifier.predict(test_df)
    
    print("\n📊 Generating figures...")
    
    # Figure 1: Confusion Matrix
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_true = le.fit_transform(test_df['risk_label'])
    y_pred = le.transform(predictions)
    viz.plot_confusion_matrix(y_true, y_pred, le.classes_)
    
    # Figure 2: Feature Importance
    feature_importance = pd.DataFrame(ml_metrics['feature_importance'])
    viz.plot_feature_importance(
        feature_importance['feature'].values,
        feature_importance['importance'].values
    )
    
    # Figure 3: Performance Comparison
    viz.plot_performance_comparison(benchmark_df)
    
    # Figure 4: ML vs Baseline
    viz.plot_ml_vs_baseline(comparison_df)
    
    # Figure 5: Risk Distribution
    viz.plot_risk_distribution(test_df, predictions)
    
    # Figure 6: Cost-Benefit Trade-off
    viz.plot_cost_benefit_analysis(comparison_df)

    print("\n🎉 All figures generated successfully!")
    print("📁 Saved to: results/figures/")


if __name__ == "__main__":
    generate_all_figures()
