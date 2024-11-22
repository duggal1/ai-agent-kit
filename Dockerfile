FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY python/ .

# Set the handler path
ENV HANDLER_PATH="/app/enterprise_ai/runpod_handler.py"

# Set the entry point
CMD [ "python", "-u", "/app/enterprise_ai/runpod_handler.py" ]