import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from .io_utils import SerializableArtifact, SklearnClassificationModel


logger = logging.getLogger('runner')


def inference_model(artifact: SerializableArtifact,
                    data: pd.DataFrame,
                    proba: bool) -> pd.Series:
  logger.info('Initiate inference, proba: %s', proba)
  model: SklearnClassificationModel = artifact.model
  pipeline: Pipeline = artifact.pipeline
  stats = artifact.stats

  logger.debug('Model stats during training...')
  for key, value in stats.items():
    logger.debug('Metric [%s]: [%.3f]', key, value)

  X = pipeline.transform(data)
  if proba:
    out = model.predict_proba(X)[:, 1]
  else:
    out = model.predict(X)

  out = pd.Series(out, name='prediction')
  return out
