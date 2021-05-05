from enum import Enum

# CATEGORICAL_POLICIES = ['label_encoding', 'one_hot', 'frequency_encoding', 'none']
# NORMALIZATION_POLICIES = ['min_max', 'standard', 'none']

class CategoricalPolicy(Enum):
  ORDINAL_ENCODING = 'ordinal_encoding'
  ONE_HOT = 'one_hot'
  FREQUENCY_ENCODING = 'frequency_encoding'
  NONE = 'none'


class NormalizationPolicy(Enum):
  MIN_MAX = 'min_max'
  STANDARD = 'standard'
  NONE = 'none'


class ModelType(Enum):
  RANDOM_FOREST = 'random_forest'
  SVC = 'svc'
  MLPClassifier = 'mlp_classifier'

