# AutoML System Built From Scratch (Finance-Grade)

## Overview
This project is a **full end-to-end AutoML (Automatic Machine Learning) system built from scratch**, without relying on any AutoML frameworks such as Auto-sklearn, H2O, or TPOT.

The goal of this project is to demonstrate **real ML systems engineering**, not just model training.  
The system automatically handles:
- Dataset profiling
- Feature engineering
- Feature selection
- Model selection
- Hyperparameter optimization
- Cross-validation
- Model persistence
- Inference
- Explainability (SHAP)

The project was tested on **real financial time-series data (Ethereum market data)**, but it is designed to work on **any tabular dataset** (classification or regression).

This project is suitable for:
- Final-year engineering projects
- Resume / portfolio projects
- Interview deep dives for ML, data science, and quant roles

---

## Key Capabilities

- Automatic detection of **classification vs regression**
- Numeric feature engineering:
  - log, sqrt, power transformations
- Datetime feature engineering:
  - hour, day, month, weekday
  - cyclical sin/cos encoding
- Feature selection:
  - Constant feature removal
  - Correlation pruning
  - Mutual information ranking
- Model families supported:
  - Logistic / Linear Regression
  - Random Forest
  - Extra Trees
  - XGBoost
  - LightGBM
  - CatBoost
  - Neural Networks (MLP)
- Bayesian hyperparameter optimization using **Optuna**
- Cross-validation with stability penalty
- Automatic best-model selection
- Model + metadata persistence
- Production-ready inference pipeline
- SHAP-based global and local explainability

---

## Project Architecture

```
automl/
├── profiler/            # Dataset analysis and problem detection
├── features/            # Feature engineering modules
├── models/              # Model definitions and trainers
├── optimization/        # Hyperparameter optimization logic
├── evaluation/          # Validation and scoring logic
├── pipeline.py          # End-to-end AutoML orchestration
├── config.py            # Global configuration
├── logger.py            # Centralized logging
automl.py                # CLI entrypoint for training
predict.py               # Inference script
explain.py               # SHAP explainability
reports/
├── best_model.pkl       # Final trained model
├── metadata.pkl         # Feature list, problem type, configs
```

---

## How the System Works (Detailed)

### 1. Dataset Profiling
- Detects problem type (classification or regression)
- Identifies numeric, categorical, and datetime features
- Computes dataset statistics

### 2. Feature Engineering
- Numeric features are expanded using mathematical transforms
- Datetime features are decomposed into meaningful components
- Cyclical time features are encoded using sin/cos transforms

### 3. Feature Selection
- Constant features are removed
- Highly correlated features are pruned
- Mutual information is used to select informative features

### 4. Model Search
- Multiple model families are trained
- All models are evaluated using cross-validation

### 5. Hyperparameter Optimization
- Optuna performs Bayesian optimization
- Each model has its own search space
- Models are penalized for unstable CV performance

### 6. Final Training
- Best model is retrained on full dataset
- Model and metadata are saved to disk

### 7. Inference
- New data is passed through the same feature engineering
- Features are aligned using metadata
- Predictions are generated safely

### 8. Explainability
- SHAP is used for model interpretability
- Global feature importance plots
- Local (per-row) explanations

---

## Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Train AutoML
```bash
python automl.py --data dataset.csv --target label
```

### Run Inference
```bash
python predict.py --model reports/best_model.pkl --meta reports/metadata.pkl --data new_data.csv --output predictions.csv
```

### Explain Model (SHAP)
```bash
python explain.py --model reports/best_model.pkl --meta reports/metadata.pkl --data dataset.csv
```

---

## Why This Project Is Different

- No AutoML black-box libraries
- Clear separation of training vs inference
- Reproducible and auditable ML pipeline
- Explainability-first design
- Modular and extensible architecture

---

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- SHAP
- Matplotlib

---

## Author
Built by **Aarav Mehta**  
https://github.com/AaravMehta-07
https://www.linkedin.com/in/aarav-mehta-16a183337/
Computer Engineering | AI / ML Systems
