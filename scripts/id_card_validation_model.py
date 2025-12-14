"""
ID Card Validation Model - Vision-based Authentication System

This script simulates a vision model (e.g., CNN/Vision Transformer) for validating
ID card authenticity in banking/payment systems. It generates synthetic features
that would typically be extracted from ID card images and trains a classifier
to detect fraudulent or tampered documents.

Use Case: Know Your Customer (KYC) verification for payment systems
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_ID_CARDS = 10000
FRAUD_RATE = 0.08  # 8% fraudulent IDs (higher than payment fraud due to KYC attempts)
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 11, 30)


def generate_id_card_features():
    """
    Generate synthetic ID card validation features that would be extracted
    by a vision model (CNN, Vision Transformer, etc.)

    These features simulate computer vision outputs analyzing:
    - Image quality
    - Document security features
    - OCR/text extraction quality
    - Face detection and quality
    - Tampering indicators
    """

    print("Generating synthetic ID card validation data...")

    id_cards = []

    for i in range(NUM_ID_CARDS):
        # Random submission date
        days_diff = (END_DATE - START_DATE).days
        submission_date = START_DATE + timedelta(days=random.randint(0, days_diff))

        # Determine if this ID is fraudulent
        is_fraudulent = random.random() < FRAUD_RATE

        # ID card type and issuing authority
        id_types = ['drivers_license', 'passport', 'national_id', 'state_id']
        id_type = random.choice(id_types)

        countries = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Spain', 'Italy']
        issuing_country = random.choice(countries)

        # === IMAGE QUALITY FEATURES (from vision model) ===
        # Legitimate IDs have better image quality
        if is_fraudulent:
            image_sharpness = np.random.normal(0.65, 0.15)  # Lower sharpness
            brightness_score = np.random.normal(0.60, 0.18)
            contrast_score = np.random.normal(0.62, 0.16)
        else:
            image_sharpness = np.random.normal(0.85, 0.08)  # Higher sharpness
            brightness_score = np.random.normal(0.82, 0.10)
            contrast_score = np.random.normal(0.84, 0.09)

        # Clip values to [0, 1]
        image_sharpness = np.clip(image_sharpness, 0, 1)
        brightness_score = np.clip(brightness_score, 0, 1)
        contrast_score = np.clip(contrast_score, 0, 1)

        # === DOCUMENT SECURITY FEATURES ===
        # Hologram detection (fraudulent IDs have poor or missing holograms)
        hologram_detected = random.random() > 0.3 if not is_fraudulent else random.random() > 0.7
        hologram_confidence = np.random.normal(0.88, 0.08) if not is_fraudulent else np.random.normal(0.45, 0.20)
        hologram_confidence = np.clip(hologram_confidence, 0, 1)

        # Microprint quality (fine text printed on ID)
        microprint_quality = np.random.normal(0.82, 0.10) if not is_fraudulent else np.random.normal(0.35, 0.18)
        microprint_quality = np.clip(microprint_quality, 0, 1)

        # UV feature detection (many IDs have UV-reactive elements)
        uv_features_detected = random.random() > 0.25 if not is_fraudulent else random.random() > 0.75
        uv_confidence = np.random.normal(0.85, 0.09) if not is_fraudulent else np.random.normal(0.30, 0.22)
        uv_confidence = np.clip(uv_confidence, 0, 1)

        # === OCR AND TEXT CONSISTENCY ===
        # OCR confidence (text extraction quality)
        ocr_confidence = np.random.normal(0.92, 0.06) if not is_fraudulent else np.random.normal(0.68, 0.18)
        ocr_confidence = np.clip(ocr_confidence, 0, 1)

        # Font consistency (fraudulent IDs may have inconsistent fonts)
        font_consistency_score = np.random.normal(0.90, 0.07) if not is_fraudulent else np.random.normal(0.58, 0.20)
        font_consistency_score = np.clip(font_consistency_score, 0, 1)

        # Text alignment quality
        text_alignment_score = np.random.normal(0.88, 0.08) if not is_fraudulent else np.random.normal(0.62, 0.17)
        text_alignment_score = np.clip(text_alignment_score, 0, 1)

        # === FACE DETECTION AND BIOMETRIC FEATURES ===
        face_detected = random.random() > 0.05  # Most IDs have faces
        face_detection_confidence = np.random.normal(0.94, 0.05) if face_detected else 0

        # Face quality metrics (resolution, lighting, positioning)
        if face_detected and not is_fraudulent:
            face_quality_score = np.random.normal(0.83, 0.10)
        elif face_detected and is_fraudulent:
            # Fraudulent IDs may have photo-swapped or low-quality faces
            face_quality_score = np.random.normal(0.58, 0.22)
        else:
            face_quality_score = 0
        face_quality_score = np.clip(face_quality_score, 0, 1)

        # Face symmetry (tampered photos may have asymmetry)
        face_symmetry_score = np.random.normal(0.87, 0.08) if face_detected and not is_fraudulent else np.random.normal(0.65, 0.18)
        face_symmetry_score = np.clip(face_symmetry_score, 0, 1)

        # === TAMPERING DETECTION ===
        # Edge consistency (fraudulent IDs may have inconsistent edges from tampering)
        edge_consistency = np.random.normal(0.89, 0.07) if not is_fraudulent else np.random.normal(0.55, 0.20)
        edge_consistency = np.clip(edge_consistency, 0, 1)

        # Shadow/lighting anomalies (photo replacements create lighting inconsistencies)
        lighting_consistency = np.random.normal(0.86, 0.09) if not is_fraudulent else np.random.normal(0.48, 0.22)
        lighting_consistency = np.clip(lighting_consistency, 0, 1)

        # Color histogram anomalies
        color_histogram_score = np.random.normal(0.84, 0.10) if not is_fraudulent else np.random.normal(0.52, 0.19)
        color_histogram_score = np.clip(color_histogram_score, 0, 1)

        # === DOCUMENT TEMPLATE MATCHING ===
        # How well does the ID match known templates for that document type
        template_match_score = np.random.normal(0.91, 0.06) if not is_fraudulent else np.random.normal(0.62, 0.19)
        template_match_score = np.clip(template_match_score, 0, 1)

        # === METADATA ===
        # Expiration status (expired IDs are suspicious)
        is_expired = random.random() < 0.15 if is_fraudulent else random.random() < 0.03

        # Document age (newer fraudulent IDs might have different characteristics)
        document_age_years = np.random.uniform(0, 10) if not is_fraudulent else np.random.uniform(0, 5)

        # === FRAUD LABELS ===
        fraud_types = [
            'photo_swap', 'fake_hologram', 'printed_copy', 'altered_text',
            'synthetic_id', 'template_mismatch', 'expired_genuine'
        ] if is_fraudulent else [None]
        fraud_type = random.choice(fraud_types)

        # Create the record
        id_card = {
            'id_card_id': f'ID{str(i).zfill(6)}',
            'submission_date': submission_date,
            'id_type': id_type,
            'issuing_country': issuing_country,

            # Image quality
            'image_sharpness': round(image_sharpness, 4),
            'brightness_score': round(brightness_score, 4),
            'contrast_score': round(contrast_score, 4),

            # Document security
            'hologram_detected': hologram_detected,
            'hologram_confidence': round(hologram_confidence, 4),
            'microprint_quality': round(microprint_quality, 4),
            'uv_features_detected': uv_features_detected,
            'uv_confidence': round(uv_confidence, 4),

            # OCR and text
            'ocr_confidence': round(ocr_confidence, 4),
            'font_consistency_score': round(font_consistency_score, 4),
            'text_alignment_score': round(text_alignment_score, 4),

            # Face and biometrics
            'face_detected': face_detected,
            'face_detection_confidence': round(face_detection_confidence, 4),
            'face_quality_score': round(face_quality_score, 4),
            'face_symmetry_score': round(face_symmetry_score, 4),

            # Tampering detection
            'edge_consistency': round(edge_consistency, 4),
            'lighting_consistency': round(lighting_consistency, 4),
            'color_histogram_score': round(color_histogram_score, 4),

            # Template matching
            'template_match_score': round(template_match_score, 4),

            # Metadata
            'is_expired': is_expired,
            'document_age_years': round(document_age_years, 2),

            # Labels
            'is_fraudulent': int(is_fraudulent),
            'fraud_type': fraud_type
        }

        id_cards.append(id_card)

    df = pd.DataFrame(id_cards)
    print(f"Generated {len(df)} ID card records")
    print(f"Fraudulent IDs: {df['is_fraudulent'].sum()} ({df['is_fraudulent'].mean()*100:.2f}%)")

    return df


def train_id_validation_model(df):
    """
    Train a Random Forest classifier to validate ID card authenticity
    based on vision model features
    """

    print("\n" + "="*70)
    print("Training ID Card Validation Model")
    print("="*70)

    # Prepare features
    feature_columns = [
        'image_sharpness', 'brightness_score', 'contrast_score',
        'hologram_detected', 'hologram_confidence', 'microprint_quality',
        'uv_features_detected', 'uv_confidence',
        'ocr_confidence', 'font_consistency_score', 'text_alignment_score',
        'face_detected', 'face_detection_confidence', 'face_quality_score',
        'face_symmetry_score', 'edge_consistency', 'lighting_consistency',
        'color_histogram_score', 'template_match_score',
        'is_expired', 'document_age_years'
    ]

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['id_type', 'issuing_country'], drop_first=False)

    # Update feature columns to include one-hot encoded columns
    categorical_cols = [col for col in df_encoded.columns if col.startswith(('id_type_', 'issuing_country_'))]
    all_features = feature_columns + categorical_cols

    X = df_encoded[all_features]
    y = df_encoded['is_fraudulent']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nDataset split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Fraud rate in training: {y_train.mean()*100:.2f}%")
    print(f"  Fraud rate in test: {y_test.mean()*100:.2f}%")

    # Train Random Forest model
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]

    # Evaluate model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n{'Model Performance':=^70}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")

    print(f"\n{'Classification Report (Test Set)':=^70}")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Legitimate', 'Fraudulent']))

    print(f"\n{'Confusion Matrix (Test Set)':=^70}")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"True Negatives (Legitimate correctly identified): {cm[0, 0]}")
    print(f"False Positives (Legitimate marked as fraud): {cm[0, 1]}")
    print(f"False Negatives (Fraud marked as legitimate): {cm[1, 0]}")
    print(f"True Positives (Fraud correctly identified): {cm[1, 1]}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'Top 15 Most Important Features':=^70}")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:.<50} {row['importance']:.4f}")

    return rf_model, X_test, y_test, y_test_pred, y_test_proba, feature_importance, df_encoded


def create_visualizations(y_test, y_test_pred, y_test_proba, feature_importance, df):
    """
    Create comprehensive visualizations for ID card validation model
    """

    print("\nCreating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title('Confusion Matrix - ID Card Validation', fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fraudulent ID Detection', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # 3. Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()

    # 4. Feature Importance (Top 12)
    ax4 = plt.subplot(2, 3, 4)
    top_features = feature_importance.head(12)
    colors = sns.color_palette('viridis', len(top_features))
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title('Top 12 Feature Importance', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()

    # 5. Fraud Score Distribution
    ax5 = plt.subplot(2, 3, 5)
    legitimate = y_test_proba[y_test == 0]
    fraudulent = y_test_proba[y_test == 1]
    plt.hist(legitimate, bins=50, alpha=0.6, label='Legitimate IDs', color='green')
    plt.hist(fraudulent, bins=50, alpha=0.6, label='Fraudulent IDs', color='red')
    plt.xlabel('Fraud Probability Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fraud Scores', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # 6. Fraud by ID Type
    ax6 = plt.subplot(2, 3, 6)
    fraud_by_type = df.groupby('id_type')['is_fraudulent'].agg(['sum', 'count'])
    fraud_by_type['fraud_rate'] = fraud_by_type['sum'] / fraud_by_type['count'] * 100
    fraud_by_type = fraud_by_type.sort_values('fraud_rate', ascending=False)
    colors = sns.color_palette('rocket', len(fraud_by_type))
    plt.bar(range(len(fraud_by_type)), fraud_by_type['fraud_rate'], color=colors)
    plt.xticks(range(len(fraud_by_type)), fraud_by_type.index, rotation=45)
    plt.ylabel('Fraud Rate (%)')
    plt.title('Fraud Rate by ID Card Type', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'id_card_validation_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    plt.close()


def save_predictions(df, rf_model):
    """
    Save predictions with fraud probabilities
    """

    print("\nSaving predictions...")

    # Prepare features for prediction
    feature_columns = [
        'image_sharpness', 'brightness_score', 'contrast_score',
        'hologram_detected', 'hologram_confidence', 'microprint_quality',
        'uv_features_detected', 'uv_confidence',
        'ocr_confidence', 'font_consistency_score', 'text_alignment_score',
        'face_detected', 'face_detection_confidence', 'face_quality_score',
        'face_symmetry_score', 'edge_consistency', 'lighting_consistency',
        'color_histogram_score', 'template_match_score',
        'is_expired', 'document_age_years'
    ]

    df_encoded = pd.get_dummies(df, columns=['id_type', 'issuing_country'], drop_first=False)
    categorical_cols = [col for col in df_encoded.columns if col.startswith(('id_type_', 'issuing_country_'))]
    all_features = feature_columns + categorical_cols

    X = df_encoded[all_features]

    # Generate predictions
    predictions = rf_model.predict(X)
    fraud_probabilities = rf_model.predict_proba(X)[:, 1]

    # Add to dataframe
    results_df = df.copy()
    results_df['predicted_fraudulent'] = predictions
    results_df['fraud_probability'] = fraud_probabilities
    results_df['prediction_correct'] = (results_df['is_fraudulent'] == results_df['predicted_fraudulent'])

    # Risk categorization
    def categorize_risk(prob):
        if prob < 0.3:
            return 'low_risk'
        elif prob < 0.6:
            return 'medium_risk'
        elif prob < 0.8:
            return 'high_risk'
        else:
            return 'critical_risk'

    results_df['risk_category'] = results_df['fraud_probability'].apply(categorize_risk)

    # Save to CSV
    output_dir = Path('data/synthetic')
    output_path = output_dir / 'id_card_validation_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Print summary statistics
    print(f"\n{'Prediction Summary':=^70}")
    print(f"Total IDs processed: {len(results_df)}")
    print(f"Correctly identified: {results_df['prediction_correct'].sum()} ({results_df['prediction_correct'].mean()*100:.2f}%)")
    print(f"\nRisk Distribution:")
    print(results_df['risk_category'].value_counts().sort_index())

    return results_df


def main():
    """
    Main execution function
    """

    print("="*70)
    print("ID CARD VALIDATION SYSTEM")
    print("Vision-based Authentication for Banking/Payment Systems")
    print("="*70)

    # Generate synthetic ID card data
    df = generate_id_card_features()

    # Save raw data
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'id_card_features.csv', index=False)
    print(f"\nRaw features saved to: {output_dir / 'id_card_features.csv'}")

    # Train model
    rf_model, X_test, y_test, y_test_pred, y_test_proba, feature_importance, df_encoded = train_id_validation_model(df)

    # Create visualizations
    create_visualizations(y_test, y_test_pred, y_test_proba, feature_importance, df)

    # Save predictions
    results_df = save_predictions(df, rf_model)

    print("\n" + "="*70)
    print("ID Card Validation System - Execution Complete")
    print("="*70)
    print("\nKey Outputs:")
    print("  1. data/synthetic/id_card_features.csv - Raw vision model features")
    print("  2. data/synthetic/id_card_validation_predictions.csv - Predictions and risk scores")
    print("  3. data/synthetic/id_card_validation_results.png - Performance visualizations")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
