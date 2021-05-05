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
class Artifact:
  model: SklearnClassificationModel
  pipeline: Pipeline
  stats: Optional[Dict[str, float]]


def dump_artifact(artifact: Artifact, path: str):
  logger.info('Dumping artifact to %s', path)
  with open(path, 'wb') as fwrite:
    pickle.dump(artifact, fwrite)


def load_artifact(path: str) -> Artifact:
  logger.info('Loading artifact from %s', path)
  with open(path, 'rb') as fread:
    artifact = pickle.load(fread)
    assert isinstance(artifact, Artifact), 'Loaded object type invalid'
    return artifact
