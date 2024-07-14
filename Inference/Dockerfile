# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set environment variables
ENV KAFKA_BROKER=${KAFKA_BROKER}
ENV NORMALIZED_DATA_TOPIC=${NORMALIZED_DATA_TOPIC}
ENV PREDICTION_TOPIC=${PREDICTION_TOPIC}

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the Flask port
EXPOSE 5001

# Command to run the Flask application
CMD ["python", "app.py"]
