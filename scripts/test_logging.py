"""Quick test for logging configuration"""
import sys
sys.path.insert(0, "D:\\AI\\OncoLink")

import os
os.chdir("D:\\AI\\OncoLink")

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
