FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p data models

# Expose port for the Flask app
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]