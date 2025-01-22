# Python official image
FROM python:3.11.9

# Stablish a folder inside the container
WORKDIR /app

# Copy necessary files into the container
COPY requirements.txt requirements.txt
COPY chat_service.py chat_service.py

# Install all the dependencies from the file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port if neccesary (in this case, only if the service executes an HTTP server)
EXPOSE 8002

# Start the application using uvicorn
CMD ["uvicorn", "chat_service:app", "--host", "0.0.0.0", "--port", "8002"]