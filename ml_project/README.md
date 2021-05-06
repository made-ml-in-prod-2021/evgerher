ml_project
==============================
[![Pytest](https://github.com/made-ml-in-prod-2021/evgerher/actions/workflows/test.yml/badge.svg)](https://github.com/made-ml-in-prod-2021/evgerher/actions/workflows/test.yml)
[![Linter](https://github.com/made-ml-in-prod-2021/evgerher/actions/workflows/linter.yml/badge.svg)](https://github.com/made-ml-in-prod-2021/evgerher/actions/workflows/linter.yml)

Dataset is taken from https://www.kaggle.com/ronitf/heart-disease-uci   
`kaggle datasets download -d ronitf/heart-disease-uci`  

**Main entry point of an app is `ml_code/runner.py`, take a look on details `--help`.**
Logs will be present in `stdout` as well as in `runner.log` file.  

## Initial analysis

Folder `notebooks` contains 2 `.ipynb` files.

- `EDA.ipynb` is a study of features relationship  
- `Models.ipynb` is a study of model performances  

## Train

Train a model with a script via next command:
`python ml_code/runner.py train --config configs/svc.yaml --output models/svc_artifact.pkl`  

It will save the artifact into `models` directory.  

### Artifact 

It is a binary format for:

- `model`: Pretrained Sklearn classification model  
- `pipeline`: Created (according to config) and fitted pipeline to be used during inference  
- `stats`: statistics about training process (quality metrics)  

## Inference

You may run an inference with next command:
`python ml_code/runner.py inference --model models/rf_artifact.pkl --data data/inference.csv --output rf_result.csv --proba`

- Data file is expected to be `csv` format without `target` column.  
- The order of features is expected to be the same as during training (otherwise not tested)  
- `proba` flag defines whether to return probabilities


## Custom configuration

1) Take a look on examples in `configs/`  
2) Consider the strict rules of allowed models within `ml_code/utils/enums.py` and `ml_code/utils/config.py`  

## Tests and linter

Run tests via `python -m pytest -v --cov .`  
Run linter via `pylint --rcfile=linter.rc ml_code/`
