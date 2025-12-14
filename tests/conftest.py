"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides shared fixtures
for all tests in the test suite.
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_transactions():
    """Sample transaction data for testing."""
    return pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
        'customer_id': [1, 2, 3],
        'amount': [100.50, 250.00, 75.25],
        'merchant_category': ['retail', 'travel', 'dining'],
        'status': ['completed', 'completed', 'failed']
    })


@pytest.fixture
def project_root():
    """Path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Path to data directory."""
    return project_root / "data"
