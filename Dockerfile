FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for any compiled python packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 10000 for Render
EXPOSE 10000

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "10000"]
