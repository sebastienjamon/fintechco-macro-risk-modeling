"""
Linear Regression Model: Revenue Prediction using Macroeconomic Indicators

This script demonstrates how to use FRED macroeconomic data to predict
FinTechCo's internal revenue using linear regression.

Model Purpose:
- Predict monthly revenue based on macroeconomic indicators
- Understand the relationship between macro trends and business performance
- Provide insights for strategic planning and risk assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load synthetic internal data and real FRED macroeconomic data.
    Merge them on date for modeling.
    """
    print("Loading data...")

    # Load internal monthly metrics
    internal_monthly = pd.read_csv('data/synthetic/monthly_internal_metrics.csv')
    internal_monthly['year_month'] = pd.to_datetime(internal_monthly['year_month'])

    # Load FRED macroeconomic data
    fed_funds = pd.read_csv('data/fred/federal_funds_rate.csv')
    fed_funds['date'] = pd.to_datetime(fed_funds['date'])

    cpi = pd.read_csv('data/fred/consumer_price_index.csv')
    cpi['date'] = pd.to_datetime(cpi['date'])

    unemployment = pd.read_csv('data/fred/unemployment_rate.csv')
    unemployment['date'] = pd.to_datetime(unemployment['date'])

    # Calculate inflation rate (year-over-year CPI change)
    cpi['inflation_rate_yoy'] = cpi['cpi_index'].pct_change(12) * 100

    # Merge FRED data
    macro_data = fed_funds.merge(cpi, on='date', how='inner')
    macro_data = macro_data.merge(unemployment, on='date', how='inner')

    # Merge with internal data
    macro_data = macro_data.rename(columns={'date': 'year_month'})
    merged_data = internal_monthly.merge(macro_data, on='year_month', how='inner')

    # Drop rows with NaN (from inflation calculation)
    merged_data = merged_data.dropna()

    print(f"✓ Loaded {len(merged_data)} months of data")
    print(f"  Date range: {merged_data['year_month'].min().date()} to {merged_data['year_month'].max().date()}")

    return merged_data


def build_linear_regression_model(data):
    """
    Build a linear regression model to predict monthly revenue.
    """
    print("\n" + "=" * 60)
    print("Building Linear Regression Model")
    print("=" * 60)

    # Define features and target
    feature_cols = [
        'federal_funds_rate_percent',
        'unemployment_rate_percent',
        'inflation_rate_yoy',
        'monthly_transaction_volume',
        'avg_churn_rate_pct'
    ]

    target_col = 'monthly_revenue_usd'

    X = data[feature_cols]
    y = data[target_col]

    print(f"\nTarget variable: {target_col}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Sample size: {len(X)} observations")

    # Split data into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
    )

    print(f"\nTrain set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "-" * 60)
    print("Model Performance")
    print("-" * 60)
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"\nTraining RMSE: ${train_rmse:,.2f}")
    print(f"Test RMSE: ${test_rmse:,.2f}")
    print(f"\nTraining MAE: ${train_mae:,.2f}")
    print(f"Test MAE: ${test_mae:,.2f}")

    # Feature importance (coefficients)
    print("\n" + "-" * 60)
    print("Feature Importance (Coefficients)")
    print("-" * 60)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)

    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']:40s}: {row['coefficient']:>12,.2f}")

    print(f"\n{'Intercept':40s}: {model.intercept_:>12,.2f}")

    # Interpretation of coefficients
    print("\n" + "-" * 60)
    print("Coefficient Interpretation")
    print("-" * 60)
    print("Positive coefficients: Variable increase -> Revenue increase")
    print("Negative coefficients: Variable increase -> Revenue decrease")
    print()

    for idx, row in feature_importance.iterrows():
        feature = row['feature']
        coef = row['coefficient']
        direction = "increases" if coef > 0 else "decreases"
        print(f"• {feature}: 1 unit increase -> ${abs(coef):,.2f} revenue {direction}")

    # Save predictions
    results_df = data.iloc[len(X_train):].copy()
    results_df['predicted_revenue'] = y_test_pred
    results_df['actual_revenue'] = y_test.values
    results_df['prediction_error'] = results_df['actual_revenue'] - results_df['predicted_revenue']
    results_df['error_percent'] = (results_df['prediction_error'] / results_df['actual_revenue'] * 100)

    results_df.to_csv('data/fred/revenue_predictions.csv', index=False)
    print(f"\n✓ Predictions saved to: data/fred/revenue_predictions.csv")

    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, feature_cols


def visualize_results(data, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred):
    """
    Create visualizations of model performance.
    """
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Actual vs Predicted (Train + Test)
    ax1 = axes[0, 0]
    ax1.scatter(y_train, y_train_pred, alpha=0.6, label='Training', color='blue')
    ax1.scatter(y_test, y_test_pred, alpha=0.6, label='Test', color='red')

    # Perfect prediction line
    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Revenue ($)', fontsize=12)
    ax1.set_ylabel('Predicted Revenue ($)', fontsize=12)
    ax1.set_title('Actual vs Predicted Revenue', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Time Series of Predictions
    ax2 = axes[0, 1]
    train_dates = data['year_month'].iloc[:len(X_train)]
    test_dates = data['year_month'].iloc[len(X_train):]

    ax2.plot(train_dates, y_train, 'b-', label='Training Actual', linewidth=2)
    ax2.plot(train_dates, y_train_pred, 'b--', label='Training Predicted', alpha=0.7)
    ax2.plot(test_dates, y_test.values, 'r-', label='Test Actual', linewidth=2)
    ax2.plot(test_dates, y_test_pred, 'r--', label='Test Predicted', alpha=0.7)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Revenue ($)', fontsize=12)
    ax2.set_title('Revenue Over Time: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Residuals Plot
    ax3 = axes[1, 0]
    residuals = y_test.values - y_test_pred
    ax3.scatter(y_test_pred, residuals, alpha=0.6, color='purple')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Revenue ($)', fontsize=12)
    ax3.set_ylabel('Residuals ($)', fontsize=12)
    ax3.set_title('Residual Plot (Test Set)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Residuals Distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residuals ($)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Residuals (Test Set)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('data/fred/linear_regression_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to: data/fred/linear_regression_results.png")

    plt.close()


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 60)
    print("FinTechCo Revenue Prediction Model")
    print("Linear Regression with Macroeconomic Indicators")
    print("=" * 60)
    print()

    # Load data
    data = load_and_prepare_data()

    # Build model
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, feature_cols = \
        build_linear_regression_model(data)

    # Visualize results
    visualize_results(data, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("• This model shows how macroeconomic factors (interest rates, unemployment,")
    print("  inflation) correlate with FinTechCo's revenue.")
    print("• Use these insights for:")
    print("  - Strategic planning during economic changes")
    print("  - Risk assessment and scenario modeling")
    print("  - Revenue forecasting under different macro conditions")
    print()


if __name__ == "__main__":
    main()
