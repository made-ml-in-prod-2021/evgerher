import logging
from flask import Flask
from flask_cors import CORS

from inference.models import FeatureException, Artifact, FeatureRestrictions, InferenceRequest, Feature

logger = logging.getLogger('inference')


restrictions = FeatureRestrictions()

def validate_request(req: InferenceRequest):
  features = restrictions.categorical_columns
  features.update(restrictions.numeric_columns)

  if len(req.values) != len(features):
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

def create_app() -> Flask:
  app = Flask(__name__)
  CORS(app)

  @app.route("/predict")
  def make_inference():
    # todo: requires Artifact object
    # todo: requires FeatureChecker object
    data: InferenceRequest = None


    return 'OK'

  @app.route('/health')
  def health_check():
    return 'OK', 200

  @app.errorhandler(FeatureException)
  def handle_bad_request(e: FeatureException):
    return e.message, 400

  return app
