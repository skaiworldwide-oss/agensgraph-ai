"""Shared infrastructure for the llama-index-agensgraph demos.

Every demo imports from here so connection wiring, dataset streaming, model
construction and console output are defined once. OpenAI is used for all
embeddings/LLM calls — nothing runs on a local model.
"""
