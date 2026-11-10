"""Verify the centralized rotating logging configuration."""
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from infrastructure.logging import setup_logging

log_path = setup_logging()
logging.getLogger(__name__).info(
    "ONCOLINK LOGGING TEST - System working correctly"
)

if not log_path.exists():
    raise SystemExit(f"Log file was not created: {log_path}")

print("Logging test passed")
print(f"Check {log_path} for the test entry")
