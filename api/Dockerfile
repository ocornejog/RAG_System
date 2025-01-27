FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service file
COPY service_calls.py .

ENV PYTHONUNBUFFERED=1
ENV COMPOSITE_SERVICE_URL=http://composite-service.rag-system:8003

EXPOSE 8000

# Change this line to use service_calls.py instead of main.py
CMD ["uvicorn", "service_calls:app", "--host", "0.0.0.0", "--port", "8000"]