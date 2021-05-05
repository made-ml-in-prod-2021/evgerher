from argparse import Namespace

import pytest

from ml_code.runner import train_callback


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
