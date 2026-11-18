"""Inspect or export saved binary scGen predictions without a native model."""

from models._native_execution import predictions

if __name__ == "__main__":
    import sys
    if any(arg.split("=", 1)[0] in ("--checkpoint", "--saved", "--inputs") for arg in sys.argv[1:]):
        from models.predict import main
        main(["--model", "scgen", *sys.argv[1:]])
    else:
        predictions("scgen")
