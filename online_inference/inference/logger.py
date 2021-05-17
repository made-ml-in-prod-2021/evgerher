import sys
import logging

DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


def setup_logger():
  logger = logging.getLogger('inference')
  formatter = logging.Formatter("%(levelname)s: %(message)s")
  stdout_handler = logging.StreamHandler(sys.stdout)
  file_handler = logging.FileHandler('log.log')

  stdout_handler.setLevel(logging.INFO)
  stdout_handler.setFormatter(formatter)
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)

  logger.setLevel(logging.DEBUG)
  logger.addHandler(file_handler)
  logger.addHandler(stdout_handler)
