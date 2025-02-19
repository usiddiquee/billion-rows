# Use Python 3.11.9 as the base image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .


# Define the command to run your application
CMD ["python", "app.py"]  # Change app.py if your entry script is different
