import logging
import pickle
from dataclasses import dataclass
from typing import Optional, Union, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

logger = logging.getLogger('runner')


SklearnClassificationModel = Union[RandomForestClassifier, MLPClassifier, SVC]


@dataclass
class Artifact:
  model: SklearnClassificationModel
  pipeline: Pipeline
  stats: Optional[Dict[str, float]]

  @classmethod
  def load_artifact(cls, path: str) -> 'Artifact':
    logger.info('Loading artifact from %s', path)
    with open(path, 'rb') as fread:
      artifact = pickle.load(fread)
      assert isinstance(artifact, Artifact), 'Loaded object type invalid'
      return artifact
