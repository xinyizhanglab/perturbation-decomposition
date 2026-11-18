"""Literal historical figure styles; no file access or training."""
CELL_LABELS = ("K562", "RPE1", "HepG2", "Jurkat")

MODEL_STYLE = {
    "Ridge 1→1": ("#149E9A", "s", False, "Ridge (1 cell line in train)"),
    "Ridge 3→1": ("#149E9A", "s", True, "Ridge (3 cell lines in train)"),
    "MLP 1→1": ("#0D7377", "^", False, "MLP (1 cell line in train)"),
    "MLP 3→1": ("#0D7377", "^", True, "MLP (3 cell lines in train)"),
    "MORPH": ("#6F5CC4", "D", True, "MORPH"),
    "STATE": ("#8A94A6", "o", True, "STATE"),
    "BioLORD": ("#D98736", "P", True, "BioLORD (3 cell lines in train)*"),
    "scGen": ("#AA5C97", "D", True, "scGen (3 cell lines in train)"),
}
