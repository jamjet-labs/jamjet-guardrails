"""Build tooling for the stage 2b injection classifier.

Importable so that `tests/test_training_data.py` can reach the licence and
contamination screens, and for no other reason. Nothing here is part of the
`jamjet-guardrails` distribution: the wheel is built from
`packages = ["src/jamjet_guardrails"]`, and `src/` never imports this tree.
"""
