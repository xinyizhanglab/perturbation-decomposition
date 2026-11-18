"""Checkpoint inference or saved-prediction access for scgen."""
import sys
from models.predict import main

if __name__ == "__main__":
    main(["--model", "scgen", *sys.argv[1:]])
