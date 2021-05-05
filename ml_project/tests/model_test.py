import os
import pickle

from ml_code.features import build_pipeline
from ml_code.models import dump_artifact, load_artifact, Artifact
from ml_code.models.train_model import create_artifact, create_model

from .utils import simple_feature_config, simple_model_config


def test_load_dump_valid(simple_feature_config, simple_model_config, tmpdir):
  path = tmpdir.join('artifact.pkl')

  model = create_model(simple_model_config)
  pipeline = build_pipeline(simple_feature_config)
  artifact: Artifact = Artifact(model, pipeline, {'dog': 10.0, 'cat': 20.12})


  dump_artifact(artifact, path)
  assert os.path.exists(path), 'Dumped file does not exist'

  loaded_artifact = load_artifact(path)
  assert pickle.dumps(artifact) == pickle.dumps(loaded_artifact), 'Artifacts do not match as binary arrays'
