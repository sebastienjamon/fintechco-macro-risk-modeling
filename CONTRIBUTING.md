# Contributing to FinTechCo Macro Risk Modeling

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Project Structure](#project-structure)

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sebastienjamon/fintechco-macro-risk-modeling.git
   cd fintechco-macro-risk-modeling
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install package in editable mode with development dependencies
   pip install -e ".[dev,test]"

   # Or install from requirements files
   pip install -r requirements/base.txt
   pip install -r requirements/dev.txt
   pip install -r requirements/test.txt
   ```

4. **Setup pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

5. **Verify installation:**
   ```bash
   pytest tests/
   ```

---

## Development Workflow

### Branch Strategy

We follow a **trunk-based development** workflow:

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit regularly:**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. **Keep your branch up to date:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Branch Naming Conventions

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

**Examples:**
- `feature/ai-artifact-detection`
- `fix/data-loader-null-handling`
- `docs/update-installation-guide`

---

## Code Standards

### Style Guidelines

We follow **PEP 8** with some modifications defined in `pyproject.toml`:

- **Line length:** 100 characters
- **Code formatter:** Black
- **Import sorting:** isort
- **Type checking:** mypy (optional but encouraged)

### Formatting Your Code

Before committing, format your code:

```bash
# Format code with Black
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Check for linting issues
flake8 src/ tests/ scripts/

# Type checking (optional)
mypy src/
```

Or use pre-commit hooks to do this automatically:

```bash
pre-commit run --all-files
```

### Code Quality Checklist

- [ ] Code is formatted with Black
- [ ] Imports are sorted with isort
- [ ] No flake8 warnings
- [ ] Docstrings for all public functions and classes
- [ ] Type hints where appropriate
- [ ] Comments explain "why", not "what"

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/fintechco --cov-report=html

# Run specific test file
pytest tests/unit/test_data_loaders.py

# Run tests matching a pattern
pytest -k "test_fraud"

# Run tests in parallel (faster)
pytest -n auto
```

### Writing Tests

- **Location:** Place tests in `tests/unit/` or `tests/integration/`
- **Naming:** Test files must start with `test_`
- **Functions:** Test functions must start with `test_`
- **Fixtures:** Use `conftest.py` for shared fixtures

**Example test:**

```python
# tests/unit/test_data_loaders.py
from fintechco.data.loaders import load_transactions


def test_load_transactions(sample_transactions):
    """Test loading transaction data."""
    df = load_transactions("data/synthetic/payment_transactions.csv")

    assert len(df) > 0
    assert 'transaction_id' in df.columns
    assert 'amount' in df.columns
```

### Test Coverage

- **Target:** 80% code coverage minimum
- **View coverage:** Open `htmlcov/index.html` after running tests with `--cov`
- **CI will fail** if coverage drops below 70%

---

## Pull Request Process

### Before Opening a PR

1. **Ensure all tests pass:**
   ```bash
   pytest
   ```

2. **Format your code:**
   ```bash
   black src/ tests/ scripts/
   isort src/ tests/ scripts/
   ```

3. **Update documentation if needed**

4. **Add tests for new functionality**

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Coverage ≥80%

## Checklist
- [ ] Code formatted with Black
- [ ] Docstrings added
- [ ] README updated (if needed)
- [ ] No merge conflicts
```

### Code Review Process

1. **Automated checks:** CI runs tests and linting
2. **Peer review:** At least 1 approval required
3. **Lead DS review:** For architectural changes
4. **Merge:** Squash and merge to main

---

## Project Structure

### Package Organization

```
src/fintechco/
├── data/          # Data loading and processing
├── models/        # ML models
├── features/      # Feature engineering
├── utils/         # Utility functions
└── api/           # API endpoints (future)
```

### Adding a New Module

1. **Create module file:**
   ```bash
   touch src/fintechco/data/new_module.py
   ```

2. **Add docstring:**
   ```python
   """
   Module description.

   This module provides...
   """
   ```

3. **Add tests:**
   ```bash
   touch tests/unit/test_new_module.py
   ```

4. **Update __init__.py if needed:**
   ```python
   # src/fintechco/data/__init__.py
   from .new_module import NewClass
   ```

---

## Common Tasks

### Adding a New Dependency

1. **Add to pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       "new-package>=1.0.0",
   ]
   ```

2. **Or add to requirements/base.txt:**
   ```
   new-package>=1.0.0
   ```

3. **Reinstall:**
   ```bash
   pip install -e ".[dev,test]"
   ```

### Creating a New Script

1. **Create script in scripts/:**
   ```bash
   touch scripts/new_script.py
   ```

2. **Use package imports:**
   ```python
   #!/usr/bin/env python
   """Script description."""

   from fintechco.models import FraudDetector


   def main():
       # Your code here
       pass


   if __name__ == "__main__":
       main()
   ```

3. **Make it executable:**
   ```bash
   chmod +x scripts/new_script.py
   ```

---

## Documentation

### Adding Documentation

- **Code:** Use docstrings (Google style preferred)
- **README:** Update README.md for user-facing changes
- **API docs:** Use Sphinx (future)

### Docstring Example

```python
def train_fraud_model(data: pd.DataFrame, config: dict) -> FraudDetector:
    """
    Train a fraud detection model.

    Args:
        data: Transaction data with features and labels
        config: Model configuration dictionary

    Returns:
        Trained FraudDetector model

    Raises:
        ValueError: If data is empty or missing required columns

    Example:
        >>> data = load_transactions("data/fraud.csv")
        >>> model = train_fraud_model(data, {"n_estimators": 100})
    """
    pass
```

---

## Getting Help

### Resources

- **CLAUDE.md:** Guide for using Claude Code with this project
- **TEAM.md:** Team organization and sprint planning
- **README.md:** Project overview and getting started

### Communication

- **GitHub Issues:** Bug reports and feature requests
- **Pull Requests:** Code review and discussions
- **Slack:** #data-science channel (internal)

### Questions?

- Open a GitHub issue with the label `question`
- Ask in the #data-science Slack channel
- Contact the Lead Data Scientist

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🎉
