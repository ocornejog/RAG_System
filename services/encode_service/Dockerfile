# Python official image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY encode_service.py encode_service.py

# Expose port
EXPOSE 8000

# Run service
CMD ["uvicorn", "encode_service:app", "--host", "0.0.0.0", "--port", "8000"]
