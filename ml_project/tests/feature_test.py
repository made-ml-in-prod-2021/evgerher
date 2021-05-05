from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml_code.features.features import build_column_transformer, build_categorical_pipeline
from ml_code.features.resolvers import FrequencyEncoder
from ml_code.utils import FeatureConfig, CategoricalPolicy, NormalizationPolicy
from .utils import sample_data


def test_column_transformer(sample_data):
  data, categorical_cols, numeric_cols = sample_data

  feature_config = FeatureConfig(
      categorical_cols,
      numeric_cols,
      CategoricalPolicy.ORDINAL_ENCODING,
      NormalizationPolicy.NONE,
      None
    )
  column_transformer = build_column_transformer(feature_config)

  assert data['b'].isna().any(), 'At least one nan must be present in `b`'
  assert data['e'].isna().any(), 'At least one nan must be present in `e`'
  columns = categorical_cols + numeric_cols  # column transformer mixes the order
  data_after = pd.DataFrame(column_transformer.fit_transform(data), columns=columns)

  assert not data_after['b'].isna().any(), 'No nans must be present after mean imputation in `b`'
  assert not data_after['e'].isna().any(), 'No nans must be present after mean imputation in `e`'


  for label in ['a', 'c', 'd']:
    max_value = data[label].nunique() - 1 # starts from zero, thus last item will have value N - 1
    assert np.isclose(data_after[label].min(), 0.0) and np.isclose(data_after[label].max(), max_value), (
      f'Categorical values after encoding should be limited between 0 and {max_value}'
    )


@pytest.mark.parametrize(
    ['values'],
    [
        pytest.param(['A', 'B', 'E'], id='Text column'),
        pytest.param([0, 1, 13, 42], id='Integer column'),
    ]
)
def test_frequency_encoder(values):
  column = np.random.choice(values, 100)
  column = pd.Series(column, name='test_col')
  train_freqs = defaultdict(lambda: 0)
  X_train, X_test = train_test_split(column, test_size=0.2)
  for val in X_train.values:
    train_freqs[val] += 1.0

  for key in values:
    train_freqs[key] /= len(X_train)

  policy = CategoricalPolicy.FREQUENCY_ENCODING
  pipeline = build_categorical_pipeline(policy)
  assert isinstance(pipeline[0], FrequencyEncoder), f'Expected frequency encoder, found: {pipeline[0]}'

  X_train_after = pipeline.fit_transform(X_train)
  X_test_after = pipeline.transform(X_test)

  for idx, val in X_train.iteritems():
    value_mapped = train_freqs[val]
    value_found = X_train_after.loc[idx]
    assert np.isclose(value_mapped, value_found), 'The mapping is invalid'

  for idx, val in X_test.iteritems():
    value_mapped = train_freqs[val]
    value_found = X_test_after.loc[idx]
    assert np.isclose(value_mapped, value_found), 'The mapping is invalid'


def test_frequency_encoder_dataframe(sample_data):
  data, categorical_cols, numeric_cols = sample_data
  feature_config = FeatureConfig(
    categorical_cols,
    numeric_cols,
    CategoricalPolicy.FREQUENCY_ENCODING,
    NormalizationPolicy.MIN_MAX,
    None
  )

  column_freqs = defaultdict(lambda: defaultdict(lambda: 0))
  for col in categorical_cols:
    series = data[col]
    for val in series.values:
      column_freqs[col][val] += 1.0

  for key, options in column_freqs.items():
    for option in options.keys():
      column_freqs[key][option] /= len(data)


  column_transformer = build_column_transformer(feature_config)
  columns = categorical_cols + numeric_cols  # column transformer mixes the order
  data_after = pd.DataFrame(column_transformer.fit_transform(data), columns=columns)

  for col in categorical_cols:
    column_before = data[col]
    column_after = data_after[col]
    freqs = column_freqs[col]
    for idx, value in column_before.iteritems():
      value_mapped = freqs[value]
      value_found = column_after.loc[idx]
      assert np.isclose(value_found, value_mapped), 'The mapping is invalid'
