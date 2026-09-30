"""Tier B training: a residual decision head on top of a frozen backbone's label logits.

Data flows through one record shape (`lars.train.data.Record`) regardless of source.
See `src/lars/train/README.md` for the recipe and the reasoning.
"""
