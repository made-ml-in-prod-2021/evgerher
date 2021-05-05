import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml_code.features.resolvers import resolve_categorical_policy, resolve_normalization_policy
from ml_code.utils import FeatureConfig, CategoricalPolicy, NormalizationPolicy

logger = logging.getLogger('runner')


def build_categorical_pipeline(policy: CategoricalPolicy) -> Pipeline:
  logger.info('Building categorical pipeline, policy: %s', policy.value)
  method = resolve_categorical_policy(policy)

  pipeline = Pipeline(
    [(policy.value, method)]
  )

  return pipeline


def build_numeric_pipeline() -> Pipeline:
  logger.info('Building numeric pipeline')
  pipeline = Pipeline(
    [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")), ]
  )

  return pipeline


def build_normalization_pipeline(policy: NormalizationPolicy) -> Pipeline:
  logger.info('Building normalization pipeline, policy: %s', policy.value)
  method = resolve_normalization_policy(policy)
  pipeline = Pipeline(
    [(policy.value, method)]
  )

  return pipeline


def build_pipeline(feature_config: FeatureConfig) -> Pipeline:
  column_transformer = ColumnTransformer(
    [
      (
        'categorical',
        build_categorical_pipeline(feature_config.categorical_policy),
        feature_config.categorical_columns),
      (
        'numeric',
         build_numeric_pipeline(),
         feature_config.numeric_columns
      )
    ]
  )

  pipeline = Pipeline(
    [
      (
        'column_transformer',
       column_transformer
      ),
      (
        'normalization',
        build_normalization_pipeline(feature_config.normalization_policy)
      )
    ]
  )

  return pipeline


def fit_pipeline(df: pd.DataFrame, targets: Optional[pd.Series], pipeline: Pipeline) -> Pipeline:
  return pipeline.fit(df, targets)


def convert_features(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
  converted = pipeline.transform(df)
  return converted


def extract_target(df: pd.DataFrame, target_column: str) -> pd.Series:
  assert target_column in df.columns, f'Could not find target column [{target_column}] in dataframe'
  target = df[target_column]
  return target
