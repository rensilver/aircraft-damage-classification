"""TensorFlow environment flags.

Import this module *before* importing ``tensorflow`` or ``keras``; the variables
below are only read at TensorFlow import time. Mirrors the notebook's setup cell.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
