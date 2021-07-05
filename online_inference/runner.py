import signal
import sys
import time

from ml_code.server import app
from ml_code.utils import setup_logger


def dynamite(signum, frame):
  sys.exit("die die dynamite!")

if __name__ == '__main__':
  setup_logger()
  time.sleep(20)  # preparing


  signal.signal(signal.SIGALRM, dynamite)
  signal.alarm(60) # planting the bomb
  app.run(host='0.0.0.0', port=5050)
