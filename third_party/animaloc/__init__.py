"""Minimal package initialiser written for this project.

Upstream's ``animaloc/__init__.py`` eagerly imports data, datasets, eval,
train and vizual. None of those are vendored here, so this file deliberately
imports nothing. Every other file under ``animaloc/`` is upstream's, verbatim.
"""
