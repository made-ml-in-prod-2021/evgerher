import logging
from typing import Tuple, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from ml_code.data import read_data, split_data
from ml_code.features import build_pipeline, convert_features, extract_target, fit_pipeline
from ml_code.utils import PipelineConfig, ModelConfig, ModelType, FeatureConfig, SplitConfig
from .io_utils import Artifact, SklearnClassificationModel

logger = logging.getLogger('runner')


def cross_validate_model(X: pd.DataFrame,
                         y: pd.Series,
                         model: SklearnClassificationModel,
                         cv: KFold) -> Dict[str, float]:
  logger.info('Initiate cross validation with k=%d', cv.n_splits)
  metrics = ('accuracy', 'f1', 'roc_auc', 'precision', 'recall')
  cv_results = cross_validate(
      model,
      X,
      y,
      cv=cv,
      scoring=metrics,
      return_train_score=True,
      n_jobs=-1
  )

  results = {}
  for side in ('train', 'test'):
    for metric in metrics:
      label = f'{side}_{metric}'
      mean_label = f'cv_{label}_mean'
      std_label = f'cv_{label}_std'

      mean_value = cv_results[label].mean()
      std_value = cv_results[label].std()

      results[mean_label] = mean_value
      results[std_label] = std_value

      logger.info('Metric [%s]: [%.3f]', mean_label, mean_value)
      logger.info('Metric [%s]: [%.3f]', std_label, std_value)

  assert len(results) == 2 * len(metrics) * 2, (
      f'Expected (mean, std) values for {metrics} for (train, test); found only {len(results)}'
    )

  logger.info('Finished cross validation')
  logger.info('-------------------------')

  return results


def evaluate_model(model: SklearnClassificationModel,
                   X_train,
                   Y_train,
                   X_val,
                   Y_val) -> Dict[str, float]:
  logger.info('Start evaluating a model')
  y_train_pred = model.predict(X_train)
  y_val_pred = model.predict(X_val)

  metric_names = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
  metric_methods = [roc_auc_score, accuracy_score, precision_score, recall_score, f1_score]

  results = {}

  for side, pred, expected in zip(['train', 'val'], [y_train_pred, y_val_pred], [Y_train, Y_val]):
    for metric_name, metric in zip(metric_names, metric_methods):
      value = metric(expected, pred)
      label = f'{side}_{metric_name}'
      results[label] = value
      logger.info('Metric [%s]: [%.3f]', label, value)

  logger.info('Finished evaluating a model')
  logger.info('-------------------------')
  return results


def create_model(model_config: ModelConfig) -> SklearnClassificationModel:
  if model_config.model_type is ModelType.RANDOM_FOREST:
    return RandomForestClassifier(**model_config.model_args)
  elif model_config.model_type is ModelType.SVC:
    return SVC(**model_config.model_args)
  elif model_config.model_type is ModelType.MLP_CLASSIFIER:
    return MLPClassifier(**model_config.model_args)
  raise NotImplementedError('Unknown model type', model_config.model_type)


def train_model(config: PipelineConfig) -> Artifact:
  model, pipeline = create_artifact(config.model_config, config.feature_config)

  data: pd.DataFrame = read_data(config.data_path)
  split_config: SplitConfig = config.split_config
  df_train, df_val = split_data(split_config, data)

  target_column = config.feature_config.target_column
  Y_train = extract_target(df_train, target_column)
  Y_val = extract_target(df_val, target_column)

  df_train.drop([target_column], axis=1, inplace=True)
  df_val.drop([target_column], axis=1, inplace=True)

  pipeline = fit_pipeline(df_train, Y_train, pipeline)
  X_train = convert_features(df_train, pipeline)
  X_val = convert_features(df_val, pipeline)


  if split_config.folds:
    cv = KFold(split_config.folds)
    cv_results: Dict[str, float] = cross_validate_model(X_train, Y_train, model, cv)

  model = model.fit(X_train, Y_train)
  results: Dict[str, float] = evaluate_model(model, X_train, Y_train, X_val, Y_val)

  if split_config.folds:
    results.update(cv_results)

  return Artifact(model, pipeline, results)


def create_artifact(model_config: ModelConfig,
                    feature_config: FeatureConfig) -> Tuple[SklearnClassificationModel, Pipeline]:
  model: SklearnClassificationModel = create_model(model_config)
  logger.info('Loaded model type %s with args %s',
              model_config.model_type.value,
              model_config.model_args)

  pipeline: Pipeline = build_pipeline(feature_config)
  logger.info('Loaded pipeline')

  return (model, pipeline)
