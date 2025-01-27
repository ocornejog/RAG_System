# Use slim image for smaller size
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY encode_service.py .

# Create directory for embeddings with proper permissions
RUN mkdir -p /app/data && chmod 777 /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV EMBEDDINGS_FILE=/app/data/embeddings.json

# Expose the port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "encode_service:app", "--host", "0.0.0.0", "--port", "8000"]