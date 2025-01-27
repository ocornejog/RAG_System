FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY composite_service.py .

# Create directory for documents
RUN mkdir -p /app/documents

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENCODE_SERVICE_URL=http://encode-service.rag-system:8000
ENV QUERY_SERVICE_URL=http://query-service.rag-system:8001

# Volume for documents
VOLUME /app/documents

EXPOSE 8003

CMD ["uvicorn", "composite_service:app", "--host", "0.0.0.0", "--port", "8003"] 