"""Checkpoint inference or saved-prediction access for morph."""
import sys
from models.predict import main

if __name__ == "__main__":
    main(["--model", "morph", *sys.argv[1:]])
