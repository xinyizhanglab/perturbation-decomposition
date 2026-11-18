"""Checkpoint inference or saved-prediction access for scgpt."""
import sys
from models.predict import main

if __name__ == "__main__":
    main(["--model", "scgpt", *sys.argv[1:]])
