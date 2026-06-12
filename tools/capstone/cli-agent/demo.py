"""
Demo: CLI Agent Capstone
Usage:
    python demo.py --mock
    python demo.py --real
    python demo.py --real --new       (force new session)
    python demo.py --real --clear     (wipe all data)
    python demo.py --mock --data-dir /tmp/test-agent
"""

import argparse
import os
import sys
from pathlib import Path

from agent import run, DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="CLI Agent — Tools Vertical Capstone")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Mock mode (no API key required)")
    mode.add_argument("--real", action="store_true", help="Real mode (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--new", action="store_true", help="Force a new session (don't resume)")
    parser.add_argument("--clear", action="store_true", help="Clear all saved data and start fresh")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Data directory")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run(
        data_dir=args.data_dir,
        force_new=args.new,
        clear=args.clear,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
