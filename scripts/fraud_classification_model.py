"""
Fraud Detection Classification Model

This script demonstrates how to build a classification model to detect
fraudulent payment transactions using machine learning.

Model Purpose:
- Classify transactions as fraudulent or legitimate
- Identify key features that indicate fraud risk
- Provide fraud risk scores for real-time transaction monitoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load transaction and fraud data, prepare features for modeling.
    """
    print("Loading transaction and fraud data...")

    # Load transactions
    transactions = pd.read_csv('data/synthetic/payment_transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    # Load fraud labels
    fraud_data = pd.read_csv('data/synthetic/fraud_histories.csv')

    # Merge
    data = transactions.merge(fraud_data[['transaction_id', 'is_fraud', 'fraud_score']],
                              on='transaction_id', how='inner')

    print(f"✓ Loaded {len(data)} transactions")
    print(f"  Fraud cases: {data['is_fraud'].sum()} ({data['is_fraud'].mean()*100:.2f}%)")
    print(f"  Legitimate cases: {(1-data['is_fraud']).sum()} ({(1-data['is_fraud']).mean()*100:.2f}%)")

    return data


def engineer_features(data):
    """
    Create features for fraud detection model.
    """
    print("\nEngineering features...")

    # Time-based features
    data['hour'] = data['transaction_date'].dt.hour
    data['day_of_week'] = data['transaction_date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['is_night'] = data['hour'].between(0, 6).astype(int)

    # Amount-based features
    data['amount_log'] = np.log1p(data['amount'])

    # Customer aggregations
    customer_stats = data.groupby('customer_id')['amount'].agg(['mean', 'std', 'count'])
    customer_stats.columns = ['customer_avg_amount', 'customer_std_amount', 'customer_transaction_count']
    data = data.merge(customer_stats, on='customer_id', how='left')

    # Amount deviation from customer average
    data['amount_deviation'] = (data['amount'] - data['customer_avg_amount']) / (data['customer_std_amount'] + 1)

    # Encode categorical variables
    le_type = LabelEncoder()
    le_category = LabelEncoder()
    le_method = LabelEncoder()
    le_location = LabelEncoder()
    le_device = LabelEncoder()
    le_status = LabelEncoder()

    data['transaction_type_encoded'] = le_type.fit_transform(data['transaction_type'])
    data['merchant_category_encoded'] = le_category.fit_transform(data['merchant_category'])
    data['payment_method_encoded'] = le_method.fit_transform(data['payment_method'])
    data['location_encoded'] = le_location.fit_transform(data['location'])
    data['device_type_encoded'] = le_device.fit_transform(data['device_type'])
    data['status_encoded'] = le_status.fit_transform(data['status'])

    print(f"✓ Created {len(data.columns)} features")

    return data


def build_fraud_detection_model(data):
    """
    Build a Random Forest classifier for fraud detection.
    """
    print("\n" + "=" * 60)
    print("Building Fraud Detection Model (Random Forest)")
    print("=" * 60)

    # Define features
    feature_cols = [
        'amount',
        'amount_log',
        'transaction_type_encoded',
        'merchant_category_encoded',
        'payment_method_encoded',
        'location_encoded',
        'device_type_encoded',
        'status_encoded',
        'hour',
        'day_of_week',
        'is_weekend',
        'is_night',
        'customer_avg_amount',
        'customer_transaction_count',
        'amount_deviation'
    ]

    target_col = 'is_fraud'

    # Remove rows with NaN (from customer stats with single transaction)
    data_clean = data.dropna().reset_index(drop=True)

    X = data_clean[feature_cols]
    y = data_clean[target_col]

    print(f"\nTarget variable: {target_col}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Sample size: {len(X)} transactions")

    # Split data (80/20), get indices too
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} transactions")
    print(f"  Fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Test set: {len(X_test)} transactions")
    print(f"  Fraud: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probability scores
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    print("\n" + "-" * 60)
    print("Model Performance - Training Set")
    print("-" * 60)
    print(classification_report(y_train, y_train_pred, target_names=['Legitimate', 'Fraud']))
    train_auc = roc_auc_score(y_train, y_train_proba)
    print(f"ROC-AUC Score: {train_auc:.4f}")

    print("\n" + "-" * 60)
    print("Model Performance - Test Set")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraud']))
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"ROC-AUC Score: {test_auc:.4f}")

    # Feature importance
    print("\n" + "-" * 60)
    print("Top 10 Most Important Features")
    print("-" * 60)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:.4f}")

    # Save predictions
    results_df = data_clean.iloc[test_idx].copy()
    results_df['predicted_fraud'] = y_test_pred
    results_df['fraud_probability'] = y_test_proba
    results_df['actual_fraud'] = y_test.values

    results_df.to_csv('data/synthetic/fraud_predictions.csv', index=False)
    print(f"\n✓ Predictions saved to: data/synthetic/fraud_predictions.csv")

    return model, X_train, X_test, y_train, y_test, y_train_proba, y_test_proba, feature_importance


def visualize_results(y_train, y_test, y_train_proba, y_test_proba, feature_importance):
    """
    Create visualizations for fraud detection model.
    """
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_test, (y_test_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')

    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    ax3 = axes[1, 0]
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    avg_precision = average_precision_score(y_test, y_test_proba)
    ax3.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Feature Importance
    ax4 = axes[1, 1]
    top_features = feature_importance.head(10)
    ax4.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['feature'])
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('data/synthetic/fraud_detection_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to: data/synthetic/fraud_detection_results.png")

    plt.close()


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 60)
    print("FinTechCo Fraud Detection Model")
    print("Machine Learning Classification")
    print("=" * 60)
    print()

    # Load data
    data = load_and_prepare_data()

    # Engineer features
    data = engineer_features(data)

    # Build model
    model, X_train, X_test, y_train, y_test, y_train_proba, y_test_proba, feature_importance = \
        build_fraud_detection_model(data)

    # Visualize results
    visualize_results(y_train, y_test, y_train_proba, y_test_proba, feature_importance)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("• The Random Forest model can effectively identify fraudulent transactions")
    print("• Key fraud indicators: transaction amount, time patterns, device type,")
    print("  and deviation from customer's normal behavior")
    print("• Use this model for:")
    print("  - Real-time fraud detection and prevention")
    print("  - Risk scoring for manual review prioritization")
    print("  - Understanding fraud patterns and trends")
    print()


if __name__ == "__main__":
    main()
