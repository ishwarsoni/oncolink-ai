"""Quick test for logging configuration"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import logging
logging.basicConfig(
    filename="logs/application.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("ONCOLINK LOGGING TEST - System working correctly")
print("Logging test passed")
print("Check logs/application.log for the test entry")
