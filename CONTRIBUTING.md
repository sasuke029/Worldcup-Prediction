# Contributing to World Cup Predictor

Thank you for your interest in contributing! Here's how you can help:

## Setting Up Your Development Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd worldcup-logreg
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for any new functionality in `tests/`

3. **Run tests locally**
   ```bash
   python -m pytest tests/ -v
   ```

4. **Check code quality**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new feature description"
   ```

## Pull Request Process

1. Ensure all tests pass: `python -m pytest tests/ -v`
2. Ensure code quality checks pass
3. Update README.md if needed
4. Submit PR with description of changes

## Code Style

- **Type hints**: Add type annotations to function signatures
- **Docstrings**: Use Google-style docstrings for all functions
- **Line length**: Keep lines under 100 characters
- **Formatting**: Use Black formatter

## Issues

- Check existing issues before creating new ones
- Provide clear reproduction steps for bugs
- Include Python version and dependency versions

Thank you for contributing!
