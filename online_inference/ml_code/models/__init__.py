from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Set
from abc import ABC, abstractmethod
import yaml
from marshmallow_dataclass import class_schema

from .io_utils import logger, Artifact


class FeatureException(RuntimeError):
  def __init__(self, message):
    super().__init__(message)
    self.message = message


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
  categorical_columns: Dict[str, CategoricalFeature]
  numeric_columns: Dict[str, NumericFeature]

  @classmethod
  def load_yaml(cls, path: str) -> 'FeatureRestrictions':
    with open(path, "r") as fread:
      schema = FeatureRestrictionsSchema()
      config = schema.load(yaml.safe_load(fread))
      logger.info('Loaded pipeline config from: %s', path)
      return config


FeatureRestrictionsSchema = class_schema(FeatureRestrictions)
