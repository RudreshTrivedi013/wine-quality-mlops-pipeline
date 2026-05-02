from src.logger import logging
from src.exception import CustomException
import sys

logging.info("Setup test started")

try:
    a = 1 / 0
except Exception as e:
    raise CustomException(e, sys)