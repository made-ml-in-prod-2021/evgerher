import numpy as np
import pandas as pd

from ml_code.features.features import build_column_transformer
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


