# Project Reorganization Plan

**Date:** December 14, 2025
**Reason:** Align with Python/Data Science best practices

---

## Current Issues

### 1. **Analysis Reports in Data Folder** ❌
```
data/docs/   <- Reports should not be in data/
```
**Issue:** Mixes data with analysis outputs

### 2. **No Python Package Structure** ❌
```
scripts/     <- Loose scripts, no package structure
```
**Issue:** Not installable, hard to import between modules

### 3. **Single requirements.txt** ⚠️
```
requirements.txt    <- Should be split by environment
```
**Issue:** Development dependencies mixed with production

### 4. **Missing Test Structure** ❌
```
No tests/   <- No test directory
```
**Issue:** Can't run automated tests

### 5. **Missing Essential Files** ❌
- No proper `.gitignore`
- No `setup.py` / `pyproject.toml`
- No `.env.example`
- No `CONTRIBUTING.md`
- No `LICENSE`

---

## Proposed Structure

```
fintechco-macro-risk-modeling/
├── README.md                          # ✅ Keep
├── CLAUDE.md                          # ✅ Keep
├── TEAM.md                            # ✅ Keep
├── CONTRIBUTING.md                    # 📝 NEW: Contribution guidelines
├── LICENSE                            # 📝 NEW: MIT or Apache 2.0
├── .gitignore                         # ✅ Enhance existing
├── .env.example                       # 📝 NEW: Environment variables template
├── pyproject.toml                     # 📝 NEW: Modern Python project config
├── setup.py                           # 📝 NEW: Package installation
│
├── src/                               # 📁 NEW: Source code as package
│   └── fintechco/
│       ├── __init__.py
│       ├── data/                      # Data loading and processing
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── generators.py
│       │   └── transformers.py
│       ├── models/                    # ML models
│       │   ├── __init__.py
│       │   ├── fraud_detector.py
│       │   ├── revenue_predictor.py
│       │   └── id_validator.py
│       ├── features/                  # Feature engineering
│       │   ├── __init__.py
│       │   └── extractors.py
│       ├── api/                       # API (future)
│       │   ├── __init__.py
│       │   └── endpoints.py
│       └── utils/                     # Utilities
│           ├── __init__.py
│           ├── config.py
│           └── logger.py
│
├── scripts/                           # 📝 REFACTOR: Entry point scripts only
│   ├── train_fraud_model.py          # Calls src/fintechco/models/
│   ├── train_revenue_model.py
│   ├── train_id_validator.py
│   ├── generate_data.py
│   └── fetch_fred_data.py
│
├── notebooks/                         # ✅ Keep with subdirectories
│   ├── 01_exploratory/               # 📁 NEW: EDA notebooks
│   │   └── quick_start_analysis.ipynb
│   ├── 02_modeling/                  # 📁 NEW: Model development
│   │   └── hypothesis_testing.ipynb
│   └── 03_analysis/                  # 📁 NEW: Analysis notebooks
│       ├── macro_scenario_projections.ipynb
│       └── risk_analyst_scenario_validation.ipynb
│
├── tests/                             # 📁 NEW: Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   ├── unit/                         # Unit tests
│   │   ├── test_data_loaders.py
│   │   ├── test_models.py
│   │   └── test_features.py
│   ├── integration/                  # Integration tests
│   │   └── test_full_pipeline.py
│   └── fixtures/                     # Test data
│       └── sample_data.csv
│
├── data/                              # ✅ Keep - data only
│   ├── raw/                          # 📁 NEW: Raw data (not versioned)
│   ├── processed/                    # 📁 NEW: Processed data
│   ├── fred/                         # ✅ Keep: FRED data
│   └── synthetic/                    # ✅ Keep: Synthetic data
│
├── reports/                           # 📁 NEW: Move from data/docs/
│   ├── figures/                      # PNG, PDF plots
│   └── documents/                    # Analysis markdown reports
│       ├── Macro_Risk_Assessment_DS_Input.md
│       ├── Macro_Risk_Assessment_Dec2025.md
│       └── Macro_Risk_Assessment_Dec2025_Validation.md
│
├── docs/                              # ✅ Keep - technical documentation
│   ├── PRD_AI_Generated_Document_Detection.md
│   ├── TechSpec_AI_Generated_Document_Detection.md
│   ├── api/                          # 📁 NEW: API documentation (future)
│   ├── models/                       # 📁 NEW: Model cards
│   └── architecture/                 # 📁 NEW: Architecture diagrams
│
├── config/                            # 📁 NEW: Configuration files
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── logging_config.yaml
│
├── queries/                           # ✅ Keep: SQL queries
│   └── macro_risk_assessment_queries.sql
│
├── requirements/                      # 📝 REFACTOR: Split requirements
│   ├── base.txt                      # Core dependencies
│   ├── dev.txt                       # Development tools
│   ├── test.txt                      # Testing dependencies
│   └── docs.txt                      # Documentation tools
│
├── models/                            # 📁 NEW: Trained model artifacts
│   ├── fraud_detector_v1.0.pkl
│   ├── revenue_predictor_v1.0.pkl
│   └── model_metadata.json
│
├── .github/                           # 📁 NEW: GitHub workflows
│   └── workflows/
│       ├── ci.yml                    # Run tests on PR
│       └── lint.yml                  # Code quality checks
│
└── docker/                            # 📁 NEW: Docker files (future)
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Migration Plan

### Phase 1: Essential Structure (Do Now)

1. **Create Python package structure**
   ```bash
   mkdir -p src/fintechco/{data,models,features,utils,api}
   touch src/fintechco/__init__.py
   ```

2. **Move analysis reports**
   ```bash
   mkdir -p reports/{figures,documents}
   mv data/docs/* reports/documents/
   rmdir data/docs
   ```

3. **Create test structure**
   ```bash
   mkdir -p tests/{unit,integration,fixtures}
   touch tests/__init__.py tests/conftest.py
   ```

4. **Split requirements**
   ```bash
   mkdir requirements
   # Split requirements.txt into base/dev/test/docs
   ```

5. **Add essential files**
   - `.gitignore` (enhance)
   - `pyproject.toml`
   - `setup.py`
   - `.env.example`
   - `CONTRIBUTING.md`
   - `LICENSE`

### Phase 2: Code Refactoring (Week 1-2)

1. **Refactor scripts into modules**
   - Move logic from `scripts/*.py` to `src/fintechco/`
   - Keep scripts as thin entry points

2. **Add unit tests**
   - Test data loaders
   - Test model training functions
   - Test feature engineering

3. **Organize notebooks**
   - Create subdirectories: exploratory, modeling, analysis
   - Move existing notebooks

### Phase 3: Advanced Features (Week 3-4)

1. **Configuration management**
   ```bash
   mkdir config
   # Add YAML configs for models, data pipelines
   ```

2. **CI/CD setup**
   ```bash
   mkdir -p .github/workflows
   # Add GitHub Actions for testing and linting
   ```

3. **Documentation**
   - API docs (Sphinx)
   - Model cards
   - Architecture diagrams

---

## Detailed Changes

### Change 1: Create Python Package

**Current:**
```
scripts/
├── generate_synthetic_data.py    # All logic here
├── fraud_classification_model.py # All logic here
└── ...
```

**Proposed:**
```
src/fintechco/
├── data/
│   ├── generators.py              # Data generation logic
│   └── loaders.py                 # Data loading logic
├── models/
│   └── fraud_detector.py          # Model training logic
└── ...

scripts/
├── generate_data.py               # Thin wrapper: calls src/fintechco/data/generators.py
└── train_fraud_model.py           # Thin wrapper: calls src/fintechco/models/fraud_detector.py
```

**Benefits:**
- Installable package: `pip install -e .`
- Easy imports: `from fintechco.models import FraudDetector`
- Testable modules
- Reusable across notebooks and scripts

---

### Change 2: Move Analysis Reports

**Current:**
```
data/docs/
├── Macro_Risk_Assessment_Dec2025.md
└── ...
```

**Proposed:**
```
reports/
├── documents/
│   ├── Macro_Risk_Assessment_Dec2025.md
│   └── ...
└── figures/
    ├── fraud_detection_results.png
    └── ...
```

**Benefits:**
- Clear separation: data vs. outputs
- Reports are outputs, not inputs
- Easier to find analysis results

---

### Change 3: Split Requirements

**Current:**
```
requirements.txt (all dependencies mixed)
```

**Proposed:**
```
requirements/
├── base.txt          # pandas, numpy, scikit-learn, etc.
├── dev.txt           # black, flake8, mypy, ipython
├── test.txt          # pytest, pytest-cov, pytest-mock
└── docs.txt          # sphinx, mkdocs
```

**Installation:**
```bash
# Production
pip install -r requirements/base.txt

# Development
pip install -r requirements/base.txt -r requirements/dev.txt

# Testing
pip install -r requirements/base.txt -r requirements/test.txt
```

---

### Change 4: Add Test Structure

**New:**
```
tests/
├── conftest.py                    # Pytest fixtures
├── unit/
│   ├── test_data_loaders.py
│   ├── test_fraud_model.py
│   └── test_feature_extractors.py
├── integration/
│   └── test_full_pipeline.py
└── fixtures/
    └── sample_transactions.csv
```

**Example test:**
```python
# tests/unit/test_data_loaders.py
from fintechco.data.loaders import load_transactions

def test_load_transactions(tmpdir):
    """Test transaction data loading."""
    # Create sample CSV
    sample_csv = tmpdir.join("transactions.csv")
    sample_csv.write("transaction_id,amount\n1,100.50\n")

    # Load data
    df = load_transactions(str(sample_csv))

    # Assert
    assert len(df) == 1
    assert df['amount'].iloc[0] == 100.50
```

---

### Change 5: Add pyproject.toml

**New file:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fintechco-macro-risk"
version = "0.1.0"
description = "Macro risk modeling and fraud detection for FinTechCo"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "FinTechCo Data Science Team", email = "datascience@fintechco.com"}
]
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.4.0",
    "ipython>=8.0.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

[tool.black]
line-length = 100
target-version = ['py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=src/fintechco --cov-report=html --cov-report=term"
```

---

## Benefits of Reorganization

### 1. **Professional Structure** ✅
- Follows Python packaging standards
- Easy to navigate for new team members
- Industry-standard layout

### 2. **Testability** ✅
- Clear test structure
- Pytest integration
- Code coverage tracking

### 3. **Maintainability** ✅
- Modular code (import from `src/fintechco/`)
- Separation of concerns
- Configuration management

### 4. **Collaboration** ✅
- CI/CD integration
- Code quality checks (black, flake8)
- Clear contribution guidelines

### 5. **Scalability** ✅
- Easy to add new models
- Package installable in production
- Docker-ready structure

---

## Backward Compatibility

### Existing Scripts Continue to Work

**Old way (still works):**
```bash
python scripts/fraud_classification_model.py
```

**New way (also works):**
```python
from fintechco.models import FraudDetector

model = FraudDetector()
model.train(data)
```

### Notebooks Continue to Work

**Old way (still works):**
```python
import sys
sys.path.append('../scripts')
from fraud_classification_model import train_model
```

**New way (better):**
```python
from fintechco.models import FraudDetector
```

---

## Implementation Checklist

### Phase 1: Essential (Do Now)
- [ ] Create `src/fintechco/` package structure
- [ ] Move `data/docs/` → `reports/documents/`
- [ ] Create `tests/` directory structure
- [ ] Split `requirements.txt` → `requirements/*.txt`
- [ ] Add `pyproject.toml`
- [ ] Add `setup.py`
- [ ] Enhance `.gitignore`
- [ ] Add `.env.example`
- [ ] Add `CONTRIBUTING.md`
- [ ] Add `LICENSE` (MIT recommended)

### Phase 2: Code Refactoring (Week 1-2)
- [ ] Refactor `scripts/generate_synthetic_data.py` → `src/fintechco/data/generators.py`
- [ ] Refactor `scripts/fraud_classification_model.py` → `src/fintechco/models/fraud_detector.py`
- [ ] Refactor `scripts/id_card_validation_model.py` → `src/fintechco/models/id_validator.py`
- [ ] Write unit tests for data loaders
- [ ] Write unit tests for models
- [ ] Organize notebooks into subdirectories

### Phase 3: Advanced (Week 3-4)
- [ ] Add `config/` directory with YAML configs
- [ ] Setup GitHub Actions CI/CD
- [ ] Add model cards to `docs/models/`
- [ ] Add architecture diagrams
- [ ] Setup Docker (optional)

---

## Decision

**Proceed with reorganization?**

- ✅ **Yes, full reorganization** - Implement all phases
- ⚠️ **Partial** - Only Phase 1 (essential structure)
- ❌ **No** - Keep current structure

**Recommendation:** **Phase 1 now**, Phase 2-3 as team grows

---

**Document Version:** 1.0
**Created:** December 14, 2025
**Status:** Awaiting approval
