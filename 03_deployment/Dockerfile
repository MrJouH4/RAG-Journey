# Use a lightweight python image to save internet bandwidth during download
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies without saving cache to keep container minimal
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code
COPY ./app /code/app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application using uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "app"]