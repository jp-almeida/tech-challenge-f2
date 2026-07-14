# E-commerce Recommendation System

This project contains a recommendation system for an e-commerce platform using PyTorch, Scikit-Learn, MLflow, and DVC. It adheres to Clean Code and SOLID principles.

## Getting Started

1. **Install dependencies:**
   This project uses `poetry` to manage dependencies.
   ```bash
   poetry install
   ```

2. **Setup environment:**
   Copy `.env.example` to `.env`. Validate the environment using:
   ```bash
   poetry run python scripts/validate_env.py
   ```

3. **Run the complete pipeline via Docker:**
   ```bash
   docker-compose up --build
   ```
   Docker runs `preprocess`, `feature_eng`, `train`, and `evaluate` through DVC.
   It compares a Random Forest baseline with two MLP configurations, uses early
   stopping, registers the best MLP in MLflow Production, and exposes MLflow at
   `http://localhost:5000`. Generated data, reports, and models remain local.

4. **Quality checks:**
   ```bash
   poetry run ruff check .
   poetry run pytest
   pre-commit install
   ```

## Running the Pipeline via DVC

You can reproduce the whole pipeline from preprocessing to training with:
```bash
dvc repro
```

## Structure
- `src/configs`: Environment and global settings using Pydantic.
- `src/data`: DVC stages for generating/preprocessing data and engineering features.
- `src/models`: PyTorch and Baseline models orchestrated via a Factory pattern.
- `src/training`: The main training loop wrapped with MLflow metrics tracking.
- `reports`: Evaluation metrics emitted by the DVC pipeline.
