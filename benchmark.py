#!/usr/bin/env python
"""
SIH26034 Legal Metrology Multi-Image Benchmark Evaluation Harness.
Usage:
    python benchmark.py --folder <path_to_images_folder>
    python benchmark.py --folder . --images product.jpg
    python benchmark.py --folder dataset/ --ground-truth annotations.json
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.benchmark_harness import main

if __name__ == "__main__":
    main()
