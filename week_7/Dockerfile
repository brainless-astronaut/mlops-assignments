# Start with a modern, slim Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install the Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Tell the world the app runs on port 8080
EXPOSE 8080

# The command to run the app
CMD gunicorn --bind 0.0.0.0:8080 app:app
