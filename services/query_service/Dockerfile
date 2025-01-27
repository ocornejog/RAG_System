# Use slim image for smaller size
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY query_service.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8001

# Run the service
CMD ["uvicorn", "query_service:app", "--host", "0.0.0.0", "--port", "8001"]