# Python official image
FROM python:3.11.9

# Set the working directory inside the container
WORKDIR /app

# Copy necessary files into the container
COPY requirements.txt requirements.txt
COPY query_service.py query_service.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8001

# Start the application using uvicorn
CMD ["uvicorn", "query_service:app", "--host", "0.0.0.0", "--port", "8001"]