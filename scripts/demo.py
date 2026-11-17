#!/usr/bin/env python3
"""Run the Agentic RAG Platform six-scenario demo.

Usage: PYTHONPATH=src python scripts/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragagent.demo import run_demo

if __name__ == "__main__":
    results = run_demo()
    if not results.get("all_ok"):
        sys.exit(1)
