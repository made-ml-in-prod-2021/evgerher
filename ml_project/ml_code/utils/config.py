from dataclasses import dataclass
from typing import List, Optional, Union, Dict
import logging

import yaml
from marshmallow_dataclass import class_schema

from .enums import CategoricalPolicy, NormalizationPolicy, ModelType

logger = logging.getLogger('runner')


@dataclass
class SplitConfig:
  test_ratio: float
  random_state: int
  folds: Optional[int]

  def validate_parameters(self):
    assert 0 < self.test_ratio < 1.0, 'Test ratio must have a value between (0, 1)'
    assert 0 <= self.random_state < 2**32, 'Random seed must have a value between [0, 2^32 - 1]'
    if self.folds:
      assert 1 < self.folds < 101, 'Number of folds must have a value between [2, 100]'


@dataclass
class FeatureConfig:
  categorical_columns: List[str]
  numeric_columns: List[str]
  categorical_policy: CategoricalPolicy
  normalization_policy: NormalizationPolicy
  target_column: Optional[str]

  def validate_parameters(self):
    categorical_set = set(self.categorical_columns)
    numeric_set = set(self.numeric_columns)

    intersected = categorical_set.intersection(numeric_set)
    assert len(intersected) == 0, f'Found intersection between column sets: {intersected}'


@dataclass
class ModelConfig:
  model_type: ModelType
  model_args: Dict[str, Union[str, int, float, bool, list]]

  def validate_parameters(self):
    pass


@dataclass
class PipelineConfig:
  split_config: SplitConfig
  model_config: ModelConfig
  feature_config: FeatureConfig
  data_path: str


  def validate_parameters(self):
    self.split_config.validate_parameters()
    self.model_config.validate_parameters()
    self.feature_config.validate_parameters()

  @classmethod
  def load_yaml(cls, path: str) -> 'PipelineConfig':
    with open(path, "r") as fread:
      schema = PipelineConfigSchema()
      config = schema.load(yaml.safe_load(fread))
      config.validate_parameters()
      logger.info('Loaded pipeline config from: %s', path)
      return config


PipelineConfigSchema = class_schema(PipelineConfig)
