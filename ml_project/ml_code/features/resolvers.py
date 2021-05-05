from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer, OrdinalEncoder

from ml_code.utils import CategoricalPolicy, NormalizationPolicy


def identity_mapping(x):
  return x


def resolve_categorical_policy(policy: CategoricalPolicy) -> TransformerMixin:
  if policy is CategoricalPolicy.ORDINAL_ENCODING:
    return OrdinalEncoder()
  elif policy is CategoricalPolicy.ONE_HOT:
    return OneHotEncoder()
  elif policy is CategoricalPolicy.FREQUENCY_ENCODING:
    raise NotImplementedError()  # todo: invent my frequency encoder as a lambda
  elif policy is CategoricalPolicy.NONE:
    return FunctionTransformer(identity_mapping)
  else:
    raise NotImplementedError("Unknown categorical policy", policy)


def resolve_normalization_policy(policy: NormalizationPolicy) -> TransformerMixin:
  if policy is NormalizationPolicy.MIN_MAX:
    return MinMaxScaler()
  elif policy is NormalizationPolicy.STANDARD:
    return StandardScaler()
  elif policy is NormalizationPolicy.NONE:
    return FunctionTransformer(identity_mapping)
  else:
    raise NotImplementedError("Unknown normalization policy", policy)
