import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Union
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

SklearnClassificationModel = Union[RandomForestClassifier, MLPClassifier, SVC]

logger = logging.getLogger('runner')


@dataclass
class SerializableArtifact:
  model: SklearnClassificationModel
  pipeline: Pipeline
  stats: Optional[Dict[str, float]]


def dump_artifact(artifact: SerializableArtifact, path: str):
  logger.info('Dumping artifact to %s', path)
  with open(path, 'wb') as fw:
    pickle.dump(artifact, fw)


def load_artifact(path: str) -> SerializableArtifact:
  logger.info('Loading artifact from %s', path)
  with open(path, 'rb') as fr:
    artifact = pickle.load(fr)
    assert isinstance(artifact, SerializableArtifact), 'Loaded object type invalid'
    return artifact
