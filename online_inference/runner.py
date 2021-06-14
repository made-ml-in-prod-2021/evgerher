from ml_code.server import app
from ml_code.utils import setup_logger

if __name__ == '__main__':
  setup_logger()
  app.run(host='0.0.0.0', port=5050)
