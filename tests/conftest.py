"""Shared pytest configuration.

The extraction helpers under test live in ``main.py`` at the repository root,
so the root has to be importable regardless of where pytest was invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
