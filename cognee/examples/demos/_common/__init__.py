"""Shared infrastructure for the cognee-agensgraph demo suite.

Everything the demos need to point cognee at one AgensGraph database — connection
config, OpenAI model wiring, dataset streaming, console output, and a small cost
estimator — lives here so each demo stays focused on the one cognee capability it
showcases.
"""
