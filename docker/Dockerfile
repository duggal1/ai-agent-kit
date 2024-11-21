FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Copy your project files
COPY requirements.txt .
COPY python/ ./python/

# Install dependencies
RUN pip install -r requirements.txt

# Set up the entrypoint
CMD ["python", "python/enterprise_ai/runpod_handler.py"]
