from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer, OrdinalEncoder
import pandas as pd

from ml_code.utils import CategoricalPolicy, NormalizationPolicy


def identity_mapping(x):
  return x


class FrequencyEncoder(TransformerMixin, BaseEstimator):
  def __init__(self):
    self.mapping = None

  def fit(self, X, y=None):
    # print(X.shape, X.dtype)
    if isinstance(X, pd.Series):
      freqs = X.value_counts() / len(X)
      self.mapping = freqs
    elif isinstance(X, pd.DataFrame):
      self.mapping = {}
      for col in X.columns:
        freqs = X[col].value_counts() / len(X)
        self.mapping[col] = freqs
    else:
      raise NotImplementedError()
    return self

  def transform(self, X, y=None):
    if isinstance(X, pd.Series):
      X_upd = X.map(self.mapping)
    elif isinstance(X, pd.DataFrame):
      X_upd = X.copy()
      for col, mapping in self.mapping.items():
        X_upd[col] = X_upd[col].map(mapping)
    else:
      raise NotImplementedError()
    return X_upd


def resolve_categorical_policy(policy: CategoricalPolicy) -> TransformerMixin:
  if policy is CategoricalPolicy.ORDINAL_ENCODING:
    return OrdinalEncoder()
  elif policy is CategoricalPolicy.ONE_HOT:
    return OneHotEncoder()
  elif policy is CategoricalPolicy.FREQUENCY_ENCODING:
    return FrequencyEncoder()
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
