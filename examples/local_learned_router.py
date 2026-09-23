"""Local entry point for the same hash-pinned router used by the service."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing import LocalLearnedRouter, validate_artifact

if __name__ == '__main__':
    import json
    print(json.dumps(LocalLearnedRouter().route('What is my currently selected language? Please leave it unchanged.'), indent=2))
