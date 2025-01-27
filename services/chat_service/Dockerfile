FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files and environment variables
COPY chat_service.py .
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV QUERY_SERVICE_URL=http://query-service.rag-system:8001

EXPOSE 8002

CMD ["uvicorn", "chat_service:app", "--host", "0.0.0.0", "--port", "8002"]