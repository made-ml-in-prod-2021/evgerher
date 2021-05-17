from inference.logger import setup_logger
from inference.server import app


if __name__ == '__main__':
  setup_logger()
  app.run(host='0.0.0.0', port=5050)
