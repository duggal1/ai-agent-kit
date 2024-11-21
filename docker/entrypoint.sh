#!/bin/bash

# Check if we're running in RunPod environment
if [ -n "$RUNPOD_AI_API" ]; then
    echo "Starting RunPod handler..."
    python3 enterprise_ai/runpod_handler.py
else
    echo "Starting FastAPI server..."
    python3 server.py
fi
