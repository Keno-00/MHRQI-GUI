# Contributing to MHRQI

Thank you for your interest in contributing to MHRQI! We welcome contributions from everyone.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Keno-00/MHRQI.git
   cd MHRQI
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -e .[dev]
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Workflow

1. Fork the repository and create a new branch.
2. Make your changes.
3. Ensure tests pass:
   ```bash
   pytest tests/unit
   ```
4. Commit your changes (pre-commit will run automatically).
5. Push to your fork and submit a pull request.

## Code Style

We use `ruff` for linting and formatting. Please ensure your code adheres to these standards.
