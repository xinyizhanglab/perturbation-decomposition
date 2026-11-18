"""Checkpoint inference or saved-prediction access for state."""
import sys
from models.predict import main

if __name__ == "__main__":
    main(["--model", "state", *sys.argv[1:]])
