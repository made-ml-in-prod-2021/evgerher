import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Set
from abc import ABC, abstractmethod
import logging

import yaml
from marshmallow_dataclass import class_schema
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


logger = logging.getLogger('inference')


SklearnClassificationModel = Union[RandomForestClassifier, MLPClassifier, SVC]


class FeatureException(RuntimeError):
  def __init__(self, message):
    super().__init__(message)
    self.message = message


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


@dataclass
class InferenceResponse:
  label: bool
  proba: float


@dataclass
class InferenceRequest:
  columns: List[str]
  values: List[Union[str, int, float]]


class Feature(ABC):
  @abstractmethod
  def is_valid(self, value) -> bool:
    raise NotImplementedError()


@dataclass
class CategoricalFeature(Feature):
  values: Set[int]

  def is_valid(self, value) -> bool:
    return value in self.values

@dataclass
class NumericFeature(Feature):
  type: str
  min: Optional[float]
  max: Optional[float]

  def type_check(self, value):
    value_type = type(value)
    if self.type in ['int', 'float']:
      return value_type == int or value_type == float
    else:
      return value_type == eval(self.type)

  def is_valid(self, value) -> bool:
    if not self.type_check(value):
      return False
    if self.min and not self.min <= value:
      return False
    if self.max and not value <= self.max:
      return False
    return True


@dataclass
class FeatureRestrictions:
  categorical_columns: List[CategoricalFeature]
  numeric_columns: List[NumericFeature]

  @classmethod
  def load_yaml(cls, path: str) -> 'FeatureRestrictions':
    with open(path, "r") as fread:
      schema = FeatureRestrictionsSchema()
      config = schema.load(yaml.safe_load(fread))
      logger.info('Loaded pipeline config from: %s', path)
      return config


FeatureRestrictionsSchema = class_schema(FeatureRestrictions)
