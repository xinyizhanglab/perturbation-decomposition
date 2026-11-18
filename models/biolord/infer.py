"""Checkpoint inference or saved-prediction access for biolord."""
import sys
from models.predict import main

if __name__ == "__main__":
    main(["--model", "biolord", *sys.argv[1:]])
