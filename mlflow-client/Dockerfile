
# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . /app/

# Default command can be overridden in docker-compose.yml
CMD ["python", "module/mlflow_train.py"]
