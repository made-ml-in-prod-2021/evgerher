import logging
from flask import Flask

from inference.models import FeatureException

app = Flask(__name__)
logger = logging.getLogger('inference')

@app.route("/predict")
def make_inference():
    return 'OK'

@app.route('/health')
def health_check():
  return 'OK', 200

@app.errorhandler(FeatureException)
def handle_bad_request(e: FeatureException):
    return e.message, 400
