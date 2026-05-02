import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("research.log"),
        logging.StreamHandler()
    ]
)

root_logger = logging.getLogger()
