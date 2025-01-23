FROM python:3.11.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY composite_service.py .

# Create directory for documents
RUN mkdir -p /app/documents

# Volume for documents
VOLUME /app/documents

EXPOSE 8003

CMD ["python", "composite_service.py"] 