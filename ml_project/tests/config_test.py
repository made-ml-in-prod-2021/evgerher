from ml_code.utils import PipelineConfig
from .conftest import INVALID_YAML_PATH


def test_invalid_yaml_fails():
  try:
    _ = PipelineConfig.load_yaml(INVALID_YAML_PATH)
  except AssertionError as einfo:
    reason = einfo.args[0]
    assert reason.startswith('Found intersection between column sets:')
