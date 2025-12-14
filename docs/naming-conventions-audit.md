# Naming Conventions Audit

## Current State Analysis

### ✅ Correct Naming Conventions

**Python Scripts (snake_case):**
- `scripts/fetch_fred_data.py` ✓
- `scripts/fraud_classification_model.py` ✓
- `scripts/generate_synthetic_data.py` ✓
- `scripts/id_card_validation_model.py` ✓
- `scripts/linear_regression_model.py` ✓
- `scripts/macro_stress_analysis.py` ✓
- `scripts/verify_correlations.py` ✓

**Python Packages (lowercase):**
- `src/fintechco/` ✓
- `src/fintechco/data/` ✓
- `src/fintechco/models/` ✓
- `src/fintechco/features/` ✓
- `src/fintechco/utils/` ✓
- `src/fintechco/api/` ✓

**Test Files (snake_case with test_ prefix):**
- `tests/conftest.py` ✓
- `tests/unit/` ✓
- `tests/integration/` ✓

**Notebooks (snake_case):**
- `notebooks/hypothesis_testing.ipynb` ✓
- `notebooks/macro_scenario_projections.ipynb` ✓
- `notebooks/quick_start_analysis.ipynb` ✓
- `notebooks/risk_analyst_scenario_validation.ipynb` ✓

**SQL Files (snake_case):**
- `queries/macro_risk_assessment_queries.sql` ✓

**Config Files:**
- `pyproject.toml` ✓
- `setup.py` ✓
- `.env.example` ✓
- `.gitignore` ✓

**Root Documentation (UPPERCASE for high-visibility):**
- `README.md` ✓
- `LICENSE` ✓
- `CONTRIBUTING.md` ✓
- `CLAUDE.md` ✓
- `TEAM.md` ✓

---

## ❌ Issues to Fix

### Issue 1: Documentation Files (Should use kebab-case)

**Current:**
- `docs/PRD_AI_Generated_Document_Detection.md`
- `docs/TechSpec_AI_Generated_Document_Detection.md`
- `REORGANIZATION_PLAN.md`

**Should be:**
- `docs/prd-ai-generated-document-detection.md`
- `docs/techspec-ai-generated-document-detection.md`
- `reorganization-plan.md`

**Rationale:** Documentation files should use lowercase with hyphens (kebab-case) for web-friendliness and consistency with modern conventions.

### Issue 2: Report Documents (Should use kebab-case)

**Current:**
- `reports/documents/Macro_Risk_Assessment_DS_Input.md`
- `reports/documents/Macro_Risk_Assessment_Dec2025.md`
- `reports/documents/Macro_Risk_Assessment_Dec2025_Validation.md`

**Should be:**
- `reports/documents/macro-risk-assessment-ds-input.md`
- `reports/documents/macro-risk-assessment-dec2025.md`
- `reports/documents/macro-risk-assessment-dec2025-validation.md`

**Rationale:** Report documents are still documentation and should follow kebab-case convention.

---

## Naming Convention Standards

### Python Best Practices (PEP 8)

| Type | Convention | Example |
|------|------------|---------|
| Packages | lowercase, no underscores | `fintechco`, `data` |
| Modules | snake_case | `fraud_detector.py` |
| Classes | PascalCase | `FraudDetector` |
| Functions | snake_case | `train_model()` |
| Variables | snake_case | `fraud_score` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRIES` |
| Private | _leading_underscore | `_internal_method()` |

### File and Directory Naming

| Type | Convention | Example |
|------|------------|---------|
| Python files | snake_case | `data_loader.py` |
| Test files | test_*.py | `test_fraud_detector.py` |
| Config files | lowercase + extension | `pyproject.toml`, `.env` |
| Docs (high-level) | UPPERCASE.md | `README.md`, `CONTRIBUTING.md` |
| Docs (technical) | kebab-case.md | `api-reference.md` |
| SQL files | snake_case.sql | `create_tables.sql` |
| Notebooks | snake_case.ipynb | `fraud_analysis.ipynb` |
| Data files | snake_case.csv | `payment_transactions.csv` |

### PyPI Distribution

| Type | Convention | Example |
|------|------------|---------|
| Package name | lowercase-with-hyphens | `fintechco-macro-risk` |
| Import name | lowercase_or_single | `import fintechco` |

---

## Implementation Plan

1. **Rename documentation files** in `docs/`
2. **Rename report documents** in `reports/documents/`
3. **Rename root-level files** (`REORGANIZATION_PLAN.md`)
4. **Update all references** in:
   - README.md
   - CLAUDE.md
   - TEAM.md
   - CONTRIBUTING.md
   - Other documentation files
5. **Test links** to ensure nothing is broken
6. **Commit changes** with clear message

---

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PyPA Packaging Guide](https://packaging.python.org/en/latest/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Django Documentation Style Guide](https://docs.djangoproject.com/en/dev/internals/contributing/writing-documentation/)
