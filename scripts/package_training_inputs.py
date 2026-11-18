"""Export model-specific preprocessed inputs in a CPU allocation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.training_inputs import main


if __name__ == "__main__":
    main(["export", *sys.argv[1:]])
