from argparse import Namespace
import os
import pickle

import pytest
import pandas as pd
import numpy as np

from ml_code.models import Artifact
from ml_code.runner import train_callback, inference_callback
from ml_code.utils import setup_logger

from .conftest import temp_model, INFERENCE_DATA_PATH

setup_logger()

def test_train_cli(tmpdir, caplog):
  output_path = tmpdir.join('artifact.pkl')
  config_path = 'configs/svc.yaml'


  namespace = Namespace(
    config_path = config_path,
    output_path = str(output_path)
  )
  train_callback(namespace)
  logs = caplog.messages

  # Validate important events
  pipeline_loaded_log = f'Loaded pipeline config from: {config_path}'
  model_evaluation_log = 'Finished evaluating a model'
  dump_model_log = f'Dumping artifact to {str(output_path)}'

  assert pipeline_loaded_log in logs, 'PIPELINE CONFIG LOADED; Log message not found'
  assert model_evaluation_log in logs, 'FINISH EVALUATION; Log message not found'
  assert dump_model_log in logs, 'MODEL DUMP; Log message not found'

  assert os.path.isfile(output_path), 'Dumped model file not found'

  with open(output_path, 'rb') as fread:
    loaded_artifact = pickle.load(fread)
    assert isinstance(loaded_artifact, Artifact), 'Loaded model file is not an Artifact'

    stats = loaded_artifact.stats
    assert stats is not None and len(stats) > 0, 'Artifact does not have training statistics'

    assert loaded_artifact.model is not None, 'Artifact is missing a pretrained model'
    assert loaded_artifact.pipeline is not None, 'Artifact is missing a fitted pipeline'


@pytest.mark.parametrize(
    ['proba'],
    [
        pytest.param(False, id='Disabled proba'),
        pytest.param(True, id='Enabled proba'),
    ]
)
def test_inference_cli(proba, tmpdir, temp_model):
  model_path = str(temp_model)
  output_path = tmpdir.join('predictions.csv')

  namespace = Namespace(
    model_path=model_path,
    data_path=INFERENCE_DATA_PATH,
    proba=proba,
    output_path=output_path
  )
  inference_callback(namespace)

  assert os.path.isfile(output_path)

  initial_data = pd.read_csv(INFERENCE_DATA_PATH)
  loaded_output: pd.Series = pd.read_csv(output_path, names=['target'])['target'] # a single column named 0
  dtype = loaded_output.dtype

  assert len(initial_data) == len(loaded_output), 'Initial data and inference differ in length'

  if proba:
    assert dtype == np.float64
  else:
    assert dtype == np.int64
