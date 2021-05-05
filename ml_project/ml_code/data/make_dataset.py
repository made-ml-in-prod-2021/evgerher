import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_code.utils import SplitConfig

logger = logging.getLogger('runner')


def read_data(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)
  nrows, ncols = df.shape
  logger.info('Loaded dataset from %s; rows: %d, columns: %d', path, nrows, ncols)
  return df


def split_data(params: SplitConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  logger.info('Train/test split with ratio %.3f', params.test_ratio)
  train_df, test_df = train_test_split(df,
                                       test_size=params.test_ratio,
                                       random_state=params.random_state)
  return (train_df, test_df)


def write_results(result: pd.Series, path: str):
  logger.info('Writing results to: %s', path)
  result.to_csv(path, index=False, header=False)
