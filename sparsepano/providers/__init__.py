"""Swappable pipeline building blocks (poses / depth / connectivity / completion).

Each provider is selected by a string `model` name so an experiment changes exactly one block
via a CLI flag while everything downstream stays identical -> comparable substitution results.
"""
