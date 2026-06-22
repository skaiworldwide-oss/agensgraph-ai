"""Shared infrastructure for the lightrag-agensgraph demo suite.

Everything the demos need to talk to one AgensGraph database through LightRAG —
connection/env config, OpenAI model wiring, dataset streaming, console output,
and a ``make_rag`` factory — lives here so each demo stays focused on the one
LightRAG capability it showcases.
"""
