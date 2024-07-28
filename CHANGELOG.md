## 0.6.1 (2024-07-28)

### Fix

- **models**: fix issue during evaluation of linear models trained with log transformation

### Refactor

- **train.py**: refactor argument parsers and .env variables management - add error check for invalid number of models to train
- **assets**: add storage folders for local tests

## 0.6.0 (2024-07-27)

### Feat

- **train.py**: add CLI parsers and complete training-logging workflow for the ML pipeline
- **ml_pipeline**: conclude first iteration of ml_pipeline, support for linear and gradient boosting models, OOP structure
- **train.py**: complete pipeline, fix logger, and set up script it to run it end-to-end
- **logging**: add a shared utils submodule and a logger
- **xgboost**: add support for xgboost model training with hyperparameter optimization and related tests

### Fix

- **to_categorical_dtype**: fix bug in data preprocessing and update corresponding tests
- **training**: fix issue with the linear model training and add tests

### Refactor

- **models**: move model handling to OOP
- **logger**: restructure logger implementation
- **assets**: move project resources into assets folder
- **pandas**: migrate project from polars to pandas for better interoperability

## 0.5.0 (2024-07-24)

### Feat

- **model_training**: add structure for models training - set up linear regression

## 0.4.0 (2024-07-23)

### Feat

- **data_preprocessing**: add data preprocessing submodule

## 0.3.0 (2024-07-23)

### Feat

- **ml_pipeline**: initialize pipeline data extraction

### Refactor

- reorganize root directory

## 0.2.1 (2024-07-23)

### Refactor

- **repo**: reorganize project structure

## 0.2.0 (2024-07-22)

### Feat

- clarify problem statement
- update readme with notebook
- add notebook
- add data
- add README.md with challenge instructions

### Refactor

- restructure directory tree
