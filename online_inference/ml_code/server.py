import logging
from typing import Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from ml_code.models import FeatureException, Artifact, FeatureRestrictions, InferenceRequest, Feature

logger = logging.getLogger('runner')



artifact = Artifact.load_artifact(path='artifact.pkl')
restrictions = FeatureRestrictions.load_yaml('configs/health_features.yaml')
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=['POST'])
def endpoint_predict():
  # todo: requires Artifact object
  # todo: requires FeatureChecker object
  content = request.json
  data: InferenceRequest = InferenceRequest(**content)
  validate_request(data)
  label, proba = predict(data)

  return jsonify(dict(label=label, proba=proba)), 200


@app.route('/health', methods=['GET'])
def endpoint_health():
  return 'OK', 200

@app.errorhandler(FeatureException)
def handle_bad_request(e: FeatureException):
  return e.message, 400


def predict(req: InferenceRequest) -> Tuple[bool, float]:
  X = pd.DataFrame([req.values], columns=req.columns)
  X_piped = artifact.pipeline.transform(X)
  pred = artifact.model.predict_proba(X_piped)[0, 1]
  return bool(pred > 0.5), float(pred)


def validate_request(req: InferenceRequest):
  features = restrictions.categorical_columns
  features.update(restrictions.numeric_columns)

  if len(req.values) != len(req.columns) != len(features):
    msg = f'Invalid number of features: expected {len(features)}, found: {len(req.values)}'
    logger.error(msg)
    raise FeatureException(msg)

  for name, value in zip(req.columns, req.values):
    if name not in features:
      msg = f'Unknown feature {name}'
      logger.error(msg)
      raise FeatureException(msg)
    else:
      feature: Feature = features[name]
      if not feature.is_valid(value):
        msg = f'Invalid value for feature {name}: {value}'
        logger.error(msg)
        raise FeatureException(msg)
