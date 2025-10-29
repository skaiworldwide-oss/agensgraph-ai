#!/bin/bash

echo "Running unit tests..."
uv run pytest tests/unit -v

if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi

echo ""
echo "Running integration tests..."
uv run pytest tests/integration -s

if [ $? -ne 0 ]; then
    echo "Integration tests failed!"
    exit 1
fi

echo ""
echo "All tests passed!" 