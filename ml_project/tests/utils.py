from argparse import Namespace

import pytest
import numpy as np
import pandas as pd

from ml_code.runner import train_callback
from ml_code.utils import ModelConfig, ModelType, FeatureConfig, CategoricalPolicy, NormalizationPolicy


@pytest.fixture
def temp_model(tmpdir):
  output_path = tmpdir.join('artifact.pkl')
  config_path = 'configs/svc.yaml'

  namespace = Namespace(
    config_path=config_path,
    output_path=str(output_path)
  )
  train_callback(namespace)

  return output_path

@pytest.fixture
def simple_model_config():
  config = ModelConfig(ModelType.MLP_CLASSIFIER, {})
  return config

@pytest.fixture
def simple_feature_config():
  config = FeatureConfig(
    ['a', 'b'],
    ['c', 'd'],
    CategoricalPolicy.NONE,
    NormalizationPolicy.NONE,
    target_column=None
  )
  return config


@pytest.fixture
def sample_data():
  categorical_cols = ['a', 'd', 'c']
  numeric_cols = ['b', 'e']

  N = 100

  a_options = [0, 1]
  c_options = [1, 3, 4]
  d_options = [2, 12, 22, 32, 44]

  a = np.random.choice(a_options, size=(N, 1))
  c = np.random.choice(c_options, size=(N, 1))
  d = np.random.choice(d_options, size=(N, 1))
  b = np.random.uniform(0, 10, size=(N, 1))
  e = (np.random.random((N, 1)) * 3) - 2

  bnan_mask = np.random.randint(10, size=100) == 0 # 10% are zeros
  b[bnan_mask] = np.nan

  enan_mask = np.random.randint(10, size=100) == 0  # 10% are zeros
  e[enan_mask] = np.nan

  data = np.hstack([a, b, c, d, e])
  data = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])

  return (data, categorical_cols, numeric_cols)
