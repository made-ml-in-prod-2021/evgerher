from ml_code.utils import PipelineConfig


def test_invalid_yaml_fails():
  path = 'tests/resources/invalid_intersection.yaml'
  try:
    _ = PipelineConfig.load_yaml(path)
  except AssertionError as einfo:
    reason = einfo.args[0]
    assert reason.startswith('Found intersection between column sets:')
