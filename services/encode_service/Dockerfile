# Python official image
FROM python:3.11.9

# Stablish a folder inside the container
WORKDIR /app

# Copy all the files inside the container
COPY requirements.txt requirements.txt
COPY encode_service.py encode_service.py

# Install all the dependencies from the file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port if necessary (in this case, only if the service executes an HTTP server)
EXPOSE 8000

# Default command to execute the service
CMD ["uvicorn", "encode_service:app", "--host", "0.0.0.0", "--port", "8000"]
